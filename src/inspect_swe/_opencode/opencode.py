import json
import shlex
from pathlib import Path
from textwrap import dedent
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Sequence, cast

from inspect_ai.agent import (
    Agent,
    AgentAttempts,
    AgentState,
    BridgedToolsSpec,
    agent,
    agent_with,
    sandbox_agent_bridge,
)
from inspect_ai.agent._bridge.sandbox.types import SandboxAgentBridge
from inspect_ai.agent._bridge.util import bridge_model_generate
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    GenerateFilter,
    GenerateInput,
    Model,
    ModelOutput,
)
from inspect_ai.model._model import use_model_event_sink
from inspect_ai.scorer import score
from inspect_ai.tool import (
    MCPServerConfig,
    Skill,
    ToolChoice,
    ToolInfo,
    install_skills,
    read_skills,
)
from inspect_ai.tool._mcp._config import MCPServerConfigHTTP
from inspect_ai.util import sandbox as sandbox_env
from inspect_ai.util import store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions

from inspect_swe._util._async import is_callable_coroutine
from inspect_swe._util.centaur import CentaurOptions, run_centaur
from inspect_swe._util.messages import build_user_prompt
from inspect_swe._util.sandbox import resolve_agent_cwd
from inspect_swe._util.trace import trace

from .agentbinary import ensure_opencode_setup

_ModelFilter = Callable[
    [Model, list[ChatMessage], list[ToolInfo], ToolChoice | None, GenerateConfig],
    Awaitable[ModelOutput | GenerateInput | None],
]


@agent
def opencode(
    name: str = "OpenCode",
    description: str = dedent("""
       Open-source autonomous coding agent for the terminal, capable
       of writing, testing, debugging, and iterating on code across
       multiple languages.
    """),
    system_prompt: str | None = None,
    skills: Sequence[str | Path | Skill] | None = None,
    mcp_servers: Sequence[MCPServerConfig] | None = None,
    bridged_tools: Sequence[BridgedToolsSpec] | None = None,
    centaur: bool | CentaurOptions = False,
    attempts: int | AgentAttempts = 1,
    model: str | None = None,
    model_aliases: dict[str, str | Model] | None = None,
    opencode_model: str = "anthropic/claude-sonnet-4-5",
    filter: GenerateFilter | None = None,
    retry_refusals: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    version: Literal["auto", "sandbox", "stable", "latest"] | str = "auto",
    debug: bool | None = None,
) -> Agent:
    """OpenCode agent.

    Agent that uses [OpenCode](https://github.com/anomalyco/opencode)
    running in a sandbox with Inspect model bridging.

    Use the `attempts` option to enable additional submissions if the initial
    submission(s) are incorrect (by default, no additional attempts are permitted).

    Args:
        name: Agent name (used in multi-agent systems with `as_tool()` and `handoff()`)
        description: Agent description
        system_prompt: Additional system prompt to append
        skills: Additional [skills](https://inspect.aisi.org.uk/tools-standard.html#sec-skill) to make available to the agent.
        mcp_servers: MCP servers to make available to the agent
        bridged_tools: Host-side Inspect tools to expose to the agent via MCP
        centaur: Run in 'centaur' mode, which makes OpenCode available to an Inspect `human_cli()` agent rather than running it unattended.
        attempts: Configure agent to make multiple attempts
        model: Model name to use for inspect bridge (defaults to main model for task)
        model_aliases: Optional mapping of model names to Model instances or model name strings.
            Allows using custom Model implementations (e.g., wrapped Agents) instead of standard models.
            When a model name in the mapping is referenced, the corresponding Model/string is used.
        opencode_model: OpenCode model identifier to pass to the CLI in the form
            `provider/model` (default: `"anthropic/claude-sonnet-4-5"`). The actual model
            calls still go through the Inspect bridge; this just selects which provider
            client OpenCode uses to format the request.
        filter: Filter for intercepting bridged model requests
        retry_refusals: Should refusals be retried? (pass number of times to retry)
        cwd: Working directory to run opencode within
        env: Environment variables to set for opencode
        user: User to execute opencode with
        sandbox: Optional sandbox environment name
        version: Version of opencode to use. One of:
            - "auto": Use any available version in sandbox, otherwise download latest
            - "sandbox": Use sandbox version (raises RuntimeError if not available)
            - "stable"/"latest": Download and use the latest version
            - "x.x.x": Download and use a specific version
        debug: Trace all debug output.
    """
    # resolve centaur
    if centaur is True:
        centaur = CentaurOptions()

    # resolve model
    model = f"inspect/{model}" if model is not None else "inspect"

    # resolve skills
    resolved_skills = read_skills(skills) if skills is not None else None

    # resolve attempts
    attempts = AgentAttempts(attempts) if isinstance(attempts, int) else attempts

    # determine which provider client opencode will use, so we know which
    # provider entry's baseURL to override in the config (the bridge intercepts
    # the request regardless of which provider protocol opencode picks).
    provider_id = (
        opencode_model.split("/", 1)[0] if "/" in opencode_model else "anthropic"
    )

    async def execute(state: AgentState) -> AgentState:
        if provider_id == "openai":
            _patch_openai_responses_function_call_items()

        # determine port (use new port for each execution of agent on sample)
        MODEL_PORT = "opencode_model_port"
        port = store().get(MODEL_PORT, 3000) + 1
        store().set(MODEL_PORT, port)

        async with sandbox_agent_bridge(
            state,
            model=model,
            model_aliases=model_aliases,
            filter=None,
            sandbox=sandbox,
            retry_refusals=retry_refusals,
            port=port,
            bridged_tools=bridged_tools,
        ) as bridge:
            _ignore_title_generation_state(bridge)
            bridge.filter = _make_combined_filter(
                bridge=bridge,
                filter=filter,
                serialize_tool_calls=False,
            )

            # resolve sandbox
            sbox = sandbox_env(sandbox)

            # resolve working directory (home dir if sandbox default is '/')
            agent_cwd = await resolve_agent_cwd(sbox, user, cwd)

            # install opencode and its runtime dependencies in sandbox
            opencode_binary, dependency_bin_dirs = await ensure_opencode_setup(
                sbox, version, user
            )

            # combine static mcp configs with bridged tools' mcp servers
            all_mcp_servers = list(mcp_servers or []) + list(bridge.mcp_server_configs)

            # detect sandbox home directory
            home_result = await sbox.exec(["sh", "-c", "echo $HOME"], user=user)
            sandbox_home = home_result.stdout.strip() or "/root"

            # write opencode config to redirect provider baseURL to the bridge
            # and (optionally) configure mcp servers.
            #
            # The bridge's model-proxy server registers OpenAI-compatible
            # routes (/v1/responses, /v1/chat/completions), the Anthropic
            # Messages route (/v1/messages), and Gemini routes
            # (/v1beta/models/*, /models/*). The AI SDK provider clients
            # append the API-relative path (e.g. "/messages",
            # "/chat/completions") to the configured baseURL, so we must
            # include "/v1" in the baseURL we hand to opencode.
            bridge_url = f"http://localhost:{bridge.port}"
            provider_base_url = f"{bridge_url}/v1"
            opencode_config: dict[str, Any] = {
                "$schema": "https://opencode.ai/config.json",
                "provider": {
                    provider_id: {"options": {"baseURL": provider_base_url}},
                },
            }
            if resolved_skills is not None:
                opencode_config["permission"] = {"skill": {"*": "allow"}}
            if all_mcp_servers:
                opencode_config["mcp"] = resolve_mcp_servers(all_mcp_servers)

            opencode_config_dir = f"{sandbox_home}/.config/opencode"
            opencode_config_path = f"{opencode_config_dir}/opencode.json"
            await sbox.exec(["mkdir", "-p", opencode_config_dir], user=user)
            if resolved_skills is not None:
                await install_skills(
                    resolved_skills, sbox, user, f"{opencode_config_dir}/skills"
                )
            await sbox.write_file(opencode_config_path, json.dumps(opencode_config))

            # build system prompt (opencode run takes a single positional message
            # and has no separate --system-prompt flag, so we prepend)
            system_messages = [
                m.text for m in state.messages if isinstance(m, ChatMessageSystem)
            ]
            if system_prompt is not None:
                system_messages.append(system_prompt)

            prompt, has_assistant_response = build_user_prompt(state.messages)

            if system_messages:
                combined_system = "\n\n".join(system_messages)
                prompt = f"{combined_system}\n\n{prompt}"

            # base command
            cmd = [
                opencode_binary,
                "run",
                "--model",
                opencode_model,
                "--format",
                "json",
            ]

            # add auto-approve flag only for non-centaur mode
            if centaur is False:
                cmd.append("--dangerously-skip-permissions")

            # setup agent env (add dependencies to PATH so opencode can find them)
            path = ":".join(
                [*dependency_bin_dirs, "/usr/local/bin", "/usr/bin", "/bin"]
            )
            agent_env = {
                # belt-and-braces: set per-provider base URL env vars in addition
                # to the config file. Different opencode provider clients honor
                # different env conventions; the config file is authoritative
                # but env vars don't hurt. The bridge mounts API-specific routes
                # under /v1, so anthropic/openai callers that append "/messages"
                # or "/chat/completions" land on the right handler.
                "ANTHROPIC_BASE_URL": f"{bridge_url}/v1",
                "OPENAI_BASE_URL": f"{bridge_url}/v1",
                "ANTHROPIC_API_KEY": "sk-none",
                "OPENAI_API_KEY": "sk-none",
                "OPENCODE_CONFIG": opencode_config_path,
                "PATH": path,
                "HOME": sandbox_home,
            } | (env or {})

            if centaur:
                await _run_opencode_centaur(
                    options=centaur,
                    opencode_cmd=cmd,
                    agent_env=agent_env,
                    state=state,
                )
            else:
                debug_output: list[str] = []
                agent_prompt = prompt
                attempt_count = 0

                while True:
                    agent_cmd = cmd.copy()

                    # continue previous conversation between attempts (or when
                    # the inbound state already carries an assistant turn)
                    if has_assistant_response or attempt_count > 0:
                        agent_cmd.append("--continue")

                    # add prompt as positional argument at the end
                    agent_cmd.append(agent_prompt)

                    # run agent
                    result = await sbox.exec_remote(
                        cmd=["bash", "-c", 'exec 0</dev/null; "$@"', "bash"]
                        + agent_cmd,
                        options=ExecRemoteAwaitableOptions(
                            cwd=agent_cwd,
                            env=agent_env,
                            user=user,
                            concurrency=False,
                        ),
                        stream=False,
                    )

                    if debug:
                        debug_output.append(result.stdout)
                        debug_output.append(result.stderr)

                    if not result.success:
                        cli_error_msg = _clean_opencode_error(
                            result.stdout, result.stderr
                        )
                        raise RuntimeError(
                            f"Error executing opencode agent {result.returncode}: {cli_error_msg}"
                        )

                    attempt_count += 1
                    if attempt_count >= attempts.attempts:
                        break

                    answer_scores = await score(bridge.state)
                    if attempts.score_value(answer_scores[0].value) == 1.0:
                        break

                    if callable(attempts.incorrect_message):
                        if not is_callable_coroutine(attempts.incorrect_message):
                            raise ValueError(
                                "The incorrect_message function must be async."
                            )
                        agent_prompt = await attempts.incorrect_message(
                            bridge.state, answer_scores
                        )
                    else:
                        agent_prompt = attempts.incorrect_message

                if debug:
                    debug_output.insert(0, "OpenCode Debug Output:")
                    trace("\n".join(debug_output))

        return bridge.state

    return agent_with(execute, name=name, description=description)


def _ignore_title_generation_state(bridge: SandboxAgentBridge) -> None:
    def track_state(input: list[ChatMessage], output: ModelOutput) -> _TrackStateUpdate:
        if not _is_title_generation(input):
            messages = input + [output.message]
            if len(messages) > bridge._last_message_count:
                bridge.state.messages = messages
                bridge.state.output = output
            bridge._last_message_count = len(messages)
        checkpointer = getattr(bridge, "_cp", None)
        tick = getattr(checkpointer, "tick", None)
        return _TrackStateUpdate(tick)

    bridge._track_state = track_state  # type: ignore[method-assign]


class _TrackStateUpdate:
    def __init__(self, tick: Callable[[], Awaitable[None]] | None) -> None:
        self._tick = tick

    def __await__(self) -> Any:
        if self._tick is None:
            async def noop() -> None:
                return None

            return noop().__await__()
        return self._tick().__await__()


def _is_title_generation(messages: list[ChatMessage]) -> bool:
    user_messages = [
        message for message in messages if isinstance(message, ChatMessageUser)
    ]
    return (
        len(user_messages) >= 2
        and user_messages[0].text.strip() == "Generate a title for this conversation:"
    )


def _make_combined_filter(
    *,
    bridge: SandboxAgentBridge,
    filter: GenerateFilter | None,
    serialize_tool_calls: bool,
) -> GenerateFilter:
    async def combined_filter(
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | GenerateInput | None:
        if not serialize_tool_calls:
            if filter is None:
                return None
            return await cast(_ModelFilter, filter)(
                model, messages, tools, tool_choice, config
            )

        prepared_messages = messages
        prepared_tools = tools
        prepared_tool_choice = tool_choice
        prepared_config = config
        if filter is not None:
            result = await cast(_ModelFilter, filter)(
                model, messages, tools, tool_choice, config
            )
            if isinstance(result, ModelOutput):
                return _serialize_tool_calls(result)
            if isinstance(result, GenerateInput):
                prepared_messages = result.input
                prepared_tools = result.tools
                prepared_tool_choice = result.tool_choice
                prepared_config = result.config

        prepared_config.parallel_tool_calls = False
        with bridge_model_generate(), use_model_event_sink(bridge.model_event_sink):
            output = await model.generate(
                input=prepared_messages,
                tools=prepared_tools,
                tool_choice=prepared_tool_choice,
                config=prepared_config,
            )
        return _serialize_tool_calls(output)

    return combined_filter


def _serialize_tool_calls(output: ModelOutput) -> ModelOutput:
    choices = []
    changed = False
    for choice in output.choices:
        message = choice.message
        if message.tool_calls and len(message.tool_calls) > 1:
            message = message.model_copy(update={"tool_calls": [message.tool_calls[0]]})
            choice = choice.model_copy(update={"message": message})
            changed = True
        choices.append(choice)
    if not changed:
        return output
    return output.model_copy(update={"choices": choices})


def _patch_openai_responses_function_call_items() -> None:
    import inspect_ai.agent._bridge.responses_impl as responses_impl

    if getattr(responses_impl, "_opencode_function_call_item_patch", False):
        return

    original = responses_impl.responses_output_items_from_assistant_message

    def patched_responses_output_items_from_assistant_message(
        message: Any,
        tool_namespaces: dict[str, str] | None = None,
    ) -> list[Any]:
        items = original(message, tool_namespaces)
        patched_items = []
        for item in items:
            if getattr(item, "type", None) == "function_call":
                updates: dict[str, object] = {}
                if getattr(item, "id", None) is None:
                    updates["id"] = getattr(item, "call_id")
                if getattr(item, "status", None) is None:
                    updates["status"] = "completed"
                if updates:
                    item = item.model_copy(update=updates)
            patched_items.append(item)
        return patched_items

    responses_impl.responses_output_items_from_assistant_message = (
        patched_responses_output_items_from_assistant_message
    )
    responses_impl._opencode_function_call_item_patch = True


def resolve_mcp_servers(
    mcp_servers: Sequence[MCPServerConfig],
) -> dict[str, dict[str, Any]]:
    """Build OpenCode `mcp` config block from MCP server configs.

    OpenCode expects entries keyed by server name with either:
      - {"type": "local", "command": [...], "environment": {...}}
      - {"type": "remote", "url": "...", "headers": {...}}
    """
    out: dict[str, dict[str, Any]] = {}
    for server in mcp_servers:
        config = server.model_dump(exclude={"name", "tools", "type"}, exclude_none=True)
        entry: dict[str, Any] = {"enabled": True}
        if isinstance(server, MCPServerConfigHTTP):
            entry["type"] = "remote"
            if "url" in config:
                entry["url"] = config.pop("url")
            if "headers" in config:
                entry["headers"] = config.pop("headers")
        else:
            entry["type"] = "local"
            # opencode expects the command as a single array including args
            command = config.pop("command", None)
            args = config.pop("args", None)
            if command is None:
                raise ValueError(f"Local MCP server {server.name!r} has no command")
            cmd_list = [command] if isinstance(command, str) else list(command)
            if args:
                cmd_list = cmd_list + list(args)
            entry["command"] = cmd_list
            env_block = config.pop("env", None)
            if env_block:
                entry["environment"] = env_block
        out[server.name] = entry
    return out


def _clean_opencode_error(stdout: str, stderr: str) -> str:
    """Trim OpenCode CLI output to a manageable size for error messages."""
    combined = f"{stdout}\n{stderr}".strip()
    max_len = 2000
    if len(combined) > max_len:
        combined = combined[:max_len] + "... (truncated)"
    return combined if combined else "Unknown error (no output)"


async def _run_opencode_centaur(
    options: CentaurOptions,
    opencode_cmd: list[str],
    agent_env: dict[str, str],
    state: AgentState,
) -> None:
    instructions = (
        "OpenCode:\n\n"
        " - You may also use OpenCode via the 'opencode' command.\n"
        " - Use 'opencode run --continue' if you need to resume a previous opencode session."
    )

    # build .bashrc content - only export vars needed for the opencode alias,
    # not HOME which would break human_cli (PATH is needed for node)
    centaur_env = {k: v for k, v in agent_env.items() if k != "HOME"}
    agent_env_vars = [f'export {k}="{v}"' for k, v in centaur_env.items()]
    alias_cmd = shlex.join(opencode_cmd)
    alias_cmd = "alias opencode='" + alias_cmd.replace("'", "'\\''") + "'"
    bashrc = "\n".join(agent_env_vars + ["", alias_cmd])

    await run_centaur(options, instructions, bashrc, state)
