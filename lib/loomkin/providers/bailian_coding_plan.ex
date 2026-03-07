defmodule Loomkin.Providers.BailianCodingPlan do
  @moduledoc """
  Custom ReqLLM provider for Alibaba Model Studio Coding Plan endpoint.

  This provider wraps the stock Anthropic provider but targets the
  bailian-coding-plan endpoint. It registers as `:bailian` and supports
  models like qwen3.5-plus, qwen3-max, MiniMax-M2.5, glm-*, and kimi-k2.5.

  ## Usage

  Once registered, models are addressed as `"bailian:qwen3.5-plus"`.
  The provider automatically uses the API key from `BAILIAN_API_KEY`
  environment variable or config.

  Thinking mode is enabled by default with a budget of 8192 tokens.

  ## Configuration

  Set the API key via environment variable:

      export BAILIAN_API_KEY="sk-sp-..."

  Or in `.loomkin.toml`:

      [auth.bailian]
      api_key = "sk-sp-..."

  ## Models

  - `bailian:qwen3.5-plus` - Default powerful model (1M context, 65K output)
  - `bailian:qwen3-max-2026-01-23` - Max performance (262K context)
  - `bailian:qwen3-coder-next` - Coding optimized (262K context)
  - `bailian:qwen3-coder-plus` - Coding plus (1M context)
  - `bailian:MiniMax-M2.5` - Fast smaller model (204K context)
  - `bailian:glm-5` - GLM model (202K context)
  - `bailian:glm-4.7` - Fast GLM (202K context)
  - `bailian:kimi-k2.5` - Kimi model (262K context)
  """

  use ReqLLM.Provider,
    id: :bailian,
    default_base_url: "https://coding-intl.dashscope.aliyuncs.com/apps/anthropic/v1"

  @anthropic ReqLLM.Providers.Anthropic

  @default_thinking_budget 8192

  # ── Registration ────────────────────────────────────────────────────

  @doc """
  Register this provider with ReqLLM's provider registry.
  Call during application startup.
  """
  def register! do
    ReqLLM.Providers.register!(__MODULE__)
  end

  # ── Provider callbacks ──────────────────────────────────────────────

  @impl ReqLLM.Provider
  def prepare_request(:chat, model_spec, prompt, opts) do
    with {:ok, api_key} <- fetch_api_key(),
         {:ok, model} <- resolve_model(model_spec),
         {:ok, context} <- ReqLLM.Context.normalize(prompt, opts),
         opts_with_context = Keyword.put(opts, :context, context),
         opts_with_key = Keyword.put(opts_with_context, :api_key, api_key),
         opts_with_thinking <- maybe_add_thinking(opts_with_key, model),
         {:ok, processed_opts} <-
           ReqLLM.Provider.Options.process(@anthropic, :chat, model, opts_with_thinking) do
      http_opts = Keyword.get(processed_opts, :req_http_options, [])

      default_timeout =
        if Keyword.has_key?(processed_opts, :thinking) do
          Application.get_env(:req_llm, :thinking_timeout, 300_000)
        else
          Application.get_env(:req_llm, :receive_timeout, 120_000)
        end

      timeout = Keyword.get(processed_opts, :receive_timeout, default_timeout)
      base_url = Keyword.get(processed_opts, :base_url, base_url())

      req_keys =
        supported_provider_options() ++
          [
            :context,
            :model,
            :compiled_schema,
            :temperature,
            :max_tokens,
            :api_key,
            :tools,
            :tool_choice,
            :stream,
            :thinking,
            :provider_options,
            :reasoning_effort,
            :fixture,
            :on_unsupported,
            :n,
            :receive_timeout,
            :req_http_options,
            :base_url,
            :app_referer,
            :app_title,
            :anthropic_version,
            :anthropic_beta
          ]

      request =
        Req.new(
          [
            base_url: base_url,
            url: "/v1/messages",
            method: :post,
            receive_timeout: timeout,
            pool_timeout: timeout,
            connect_options: [timeout: timeout]
          ] ++ http_opts
        )
        |> Req.Request.register_options(req_keys)
        |> Req.Request.merge_options(
          Keyword.take(processed_opts, req_keys) ++ [model: get_api_model_id(model)]
        )
        |> attach(model, processed_opts)

      {:ok, request}
    end
  end

  @impl ReqLLM.Provider
  def prepare_request(:object, model_spec, prompt, opts) do
    prepare_request(:chat, model_spec, prompt, opts)
  end

  @impl ReqLLM.Provider
  def prepare_request(operation, _model_spec, _input, _opts) do
    {:error,
     ReqLLM.Error.Invalid.Parameter.exception(
       parameter: "operation: #{inspect(operation)} not supported by #{inspect(__MODULE__)}"
     )}
  end

  @impl ReqLLM.Provider
  def attach(request, model, user_opts) do
    api_key =
      case Keyword.get(user_opts, :api_key) do
        nil ->
          case fetch_api_key() do
            {:ok, token} -> token
            {:error, _} -> raise "No API key available for Bailian. Set BAILIAN_API_KEY."
          end

        token ->
          token
      end

    extra_option_keys = [
      :model,
      :compiled_schema,
      :temperature,
      :max_tokens,
      :api_key,
      :tools,
      :tool_choice,
      :stream,
      :thinking,
      :provider_options,
      :reasoning_effort,
      :fixture,
      :on_unsupported,
      :n,
      :receive_timeout,
      :req_http_options,
      :base_url,
      :context,
      :anthropic_version,
      :anthropic_beta,
      :app_referer,
      :app_title
    ]

    anthropic_version =
      Keyword.get(user_opts, :anthropic_version, "2023-06-01")

    request
    |> Req.Request.register_options(extra_option_keys)
    |> Req.Request.put_header("content-type", "application/json")
    |> Req.Request.put_header("authorization", "Bearer #{api_key}")
    |> Req.Request.put_header("anthropic-version", anthropic_version)
    |> maybe_add_beta_header(user_opts)
    |> Req.Request.merge_options(user_opts)
    |> Req.Request.put_private(:req_llm_model, model)
    |> ReqLLM.Step.Error.attach()
    |> ReqLLM.Step.Retry.attach(user_opts)
    |> Req.Request.append_request_steps(llm_encode_body: &encode_body/1)
    |> Req.Request.append_response_steps(llm_decode_response: &decode_response/1)
    |> ReqLLM.Step.Usage.attach(model)
    |> ReqLLM.Step.Fixture.maybe_attach(model, user_opts)
  end

  @impl ReqLLM.Provider
  def encode_body(request) do
    @anthropic.encode_body(request)
  end

  @impl ReqLLM.Provider
  def decode_response({request, response}) do
    @anthropic.decode_response({request, response})
  end

  @impl ReqLLM.Provider
  def extract_usage(body, model) do
    @anthropic.extract_usage(body, model)
  end

  @impl ReqLLM.Provider
  def attach_stream(model, context, opts, _finch_name) do
    api_key =
      case Keyword.get(opts, :api_key) do
        nil ->
          case fetch_api_key() do
            {:ok, token} -> token
            {:error, _} -> raise "No API key available for Bailian. Set BAILIAN_API_KEY."
          end

        token ->
          token
      end

    {provider_options, standard_opts} = Keyword.pop(opts, :provider_options, [])
    flattened_opts = Keyword.merge(standard_opts, provider_options)

    {translated_opts, _warnings} = translate_options(:chat, model, flattened_opts)

    default_timeout =
      if Keyword.has_key?(translated_opts, :thinking) do
        Application.get_env(:req_llm, :thinking_timeout, 300_000)
      else
        Application.get_env(:req_llm, :receive_timeout, 120_000)
      end

    translated_opts = Keyword.put_new(translated_opts, :receive_timeout, default_timeout)

    base_url = ReqLLM.Provider.Options.effective_base_url(@anthropic, model, translated_opts)
    translated_opts = Keyword.put(translated_opts, :base_url, base_url)

    anthropic_version = Keyword.get(translated_opts, :anthropic_version, "2023-06-01")

    headers = [
      {"Accept", "text/event-stream"},
      {"content-type", "application/json"},
      {"authorization", "Bearer #{api_key}"},
      {"anthropic-version", anthropic_version}
    ]

    headers = headers ++ build_beta_headers(translated_opts)

    body =
      ReqLLM.Providers.Anthropic.Context.encode_request(context, %{model: get_api_model_id(model)})
      |> add_stream_options(translated_opts)

    url = "#{base_url}/v1/messages"
    finch_request = Finch.build(:post, url, headers, Jason.encode!(body))
    {:ok, finch_request}
  rescue
    error ->
      {:error,
       ReqLLM.Error.API.Request.exception(
         reason: "Failed to build Bailian stream request: #{inspect(error)}"
       )}
  end

  @impl ReqLLM.Provider
  def decode_stream_event(event, model) do
    @anthropic.decode_stream_event(event, model)
  end

  @impl ReqLLM.Provider
  def translate_options(operation, model, opts) do
    @anthropic.translate_options(operation, model, opts)
  end

  # ── Internal helpers ────────────────────────────────────────────────

  defp fetch_api_key do
    case get_api_key() do
      nil -> {:error, :no_api_key}
      key -> {:ok, key}
    end
  end

  defp get_api_key do
    System.get_env("BAILIAN_API_KEY") ||
      case Loomkin.Config.get(:auth, :bailian) do
        nil -> nil
        config when is_map(config) -> Map.get(config, :api_key)
        _ -> nil
      end
  rescue
    _ -> nil
  end

  defp resolve_model(model_spec) when is_binary(model_spec) do
    canonical =
      model_spec
      |> String.replace_prefix("bailian:", "anthropic:")

    ReqLLM.model(canonical)
  end

  defp resolve_model(model_spec), do: ReqLLM.model(model_spec)

  defp get_api_model_id(model) do
    model.provider_model_id || model.id
  end

  defp maybe_add_thinking(opts, model) do
    if thinking_enabled_for_model?(model) and !Keyword.has_key?(opts, :thinking) do
      Keyword.put(opts, :thinking, %{type: :enabled, budget_tokens: @default_thinking_budget})
    else
      opts
    end
  end

  defp thinking_enabled_for_model?(model) when is_map(model) do
    model_id = Map.get(model, :id) || Map.get(model, :provider_model_id)

    model_id in [
      "qwen3.5-plus",
      "qwen3.5-397b-a17b",
      "MiniMax-M2.5",
      "glm-5",
      "glm-4.7",
      "kimi-k2.5"
    ]
  end

  defp thinking_enabled_for_model?(_), do: false

  defp maybe_add_beta_header(request, opts) do
    provider_opts = Keyword.get(opts, :provider_options, [])
    beta = Keyword.get(provider_opts, :anthropic_beta) || Keyword.get(opts, :anthropic_beta)

    if beta do
      Req.Request.put_header(request, "anthropic-beta", Enum.join(List.wrap(beta), ","))
    else
      request
    end
  end

  defp build_beta_headers(opts) do
    provider_opts = Keyword.get(opts, :provider_options, [])
    beta = Keyword.get(provider_opts, :anthropic_beta) || Keyword.get(opts, :anthropic_beta)

    if beta do
      [{"anthropic-beta", Enum.join(List.wrap(beta), ",")}]
    else
      []
    end
  end

  defp add_stream_options(body, opts) do
    max_tokens =
      case Keyword.get(opts, :max_tokens) do
        nil -> 4096
        v -> v
      end

    body
    |> Map.put(:stream, true)
    |> Map.put(:max_tokens, max_tokens)
    |> maybe_add_thinking_to_body(opts)
    |> maybe_add_tools(opts)
  end

  defp maybe_add_thinking_to_body(body, opts) do
    case Keyword.get(opts, :thinking) do
      nil -> body
      thinking -> Map.put(body, :thinking, thinking)
    end
  end

  defp maybe_add_tools(body, opts) do
    case Keyword.get(opts, :tools) do
      nil ->
        body

      tools when is_list(tools) ->
        encoded_tools =
          Enum.map(tools, fn
            %ReqLLM.Tool{} = tool ->
              %{
                name: tool.name,
                description: tool.description,
                input_schema: tool.parameter_schema
              }

            other ->
              other
          end)

        Map.put(body, :tools, encoded_tools)

      _ ->
        body
    end
  end
end
