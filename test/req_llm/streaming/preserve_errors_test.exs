defmodule ReqLLM.Streaming.PreserveErrorsTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Error.API
  alias ReqLLM.StreamResponse.MetadataHandle
  alias ReqLLM.StreamServer
  import ExUnit.CaptureLog
  import ReqLLM.Test.StreamServerHelpers, only: [start_server: 1]

  @envelope %{"error" => %{"message" => "denied", "code" => "blocked"}, "outer" => "sentinel"}

  test "core option is validated top-level and is not injected into nonstream defaults" do
    model = ReqLLM.model!("openai:gpt-4o-mini")
    context = ReqLLM.Context.new([ReqLLM.Context.user("test")])

    for provider <- [ReqLLM.Providers.OpenAI, ReqLLM.Providers.Anthropic, ReqLLM.Providers.Google] do
      opts =
        ReqLLM.Provider.Options.process_stream!(provider, :chat, model, context,
          preserve_stream_errors: true
        )

      assert opts[:preserve_stream_errors] == true
      defaults = ReqLLM.Provider.Options.process!(provider, :chat, model, [])
      refute Keyword.has_key?(defaults, :preserve_stream_errors)
    end

    assert_raise NimbleOptions.ValidationError, fn ->
      ReqLLM.Provider.Options.process_stream!(ReqLLM.Providers.OpenAI, :chat, model, context,
        preserve_stream_errors: "true"
      )
    end
  end

  for status <- [400, 401, 422, 500],
      {kind, body} <- [
        json: Jason.encode!(@envelope),
        text: "upstream denied: full plaintext tail",
        malformed: "{\"error\":broken JSON tail",
        empty: ""
      ] do
    test "buffers #{status} #{kind} through completion and preserves terminal failure" do
      status = unquote(status)
      body = unquote(body)
      server = start_server(preserve_stream_errors: true)
      StreamServer.http_event(server, {:status, status})
      StreamServer.http_event(server, {:headers, [{"x-request-id", "test"}]})

      for <<byte <- body>> do
        StreamServer.http_event(server, {:data, <<byte>>})
        refute match?({:error, _}, :sys.get_state(server).status)
      end

      StreamServer.http_event(server, :done)
      assert {:error, %API.Request{} = error, nil} = StreamServer.next(server)
      assert error.status == status

      expected =
        case Jason.decode(body) do
          {:ok, decoded} -> decoded
          _ -> body
        end

      assert error.response_body == expected
      assert error.headers == [{"x-request-id", "test"}]
      StreamServer.http_event(server, :done)
      StreamServer.http_event(server, {:error, :closed})
      send(server, {:DOWN, make_ref(), :process, nil, :normal})
      assert {:error, ^error, nil} = StreamServer.next(server)
      assert {:error, ^error} = StreamServer.await_metadata(server)
      StreamServer.cancel(server)
    end
  end

  test "incomplete HTTP bodies report transport failure, not a complete provider response" do
    server = start_server(preserve_stream_errors: true)
    StreamServer.http_event(server, {:status, 400})
    StreamServer.http_event(server, {:data, "{\"error\":"})
    reason = %Mint.TransportError{reason: :closed}
    StreamServer.http_event(server, {:error, reason})
    StreamServer.http_event(server, :done)
    assert {:error, ^reason, nil} = StreamServer.next(server)
    StreamServer.cancel(server)
  end

  test "normal task exit without HTTP completion does not promote a partial body" do
    server = start_server(preserve_stream_errors: true)
    StreamServer.http_event(server, {:status, 500})
    StreamServer.http_event(server, {:data, "partial"})
    send(server, {:DOWN, make_ref(), :process, nil, :normal})
    assert {:error, {:incomplete_http_response, :normal}, nil} = StreamServer.next(server)
    StreamServer.cancel(server)
  end

  test "status of a new retry attempt clears stale headers" do
    server = start_server(preserve_stream_errors: true)
    StreamServer.http_event(server, {:status, 500})
    StreamServer.http_event(server, {:headers, [{"old", "value"}]})
    StreamServer.http_event(server, {:status, 401})
    StreamServer.http_event(server, :done)

    assert {:error, %API.Request{status: 401, headers: [], response_body: ""}, nil} =
             StreamServer.next(server)

    StreamServer.cancel(server)
  end

  for terminator <- ["\n\n", ""] do
    test "SSE error preserves envelope and fails even with terminator #{inspect(terminator)}" do
      server = start_server(preserve_stream_errors: true)
      StreamServer.http_event(server, {:status, 200})

      StreamServer.http_event(
        server,
        {:data, "data: " <> Jason.encode!(@envelope) <> unquote(terminator)}
      )

      StreamServer.http_event(server, :done)

      assert {:error, %API.StreamEvent{status: 200, response_body: @envelope}, nil} =
               StreamServer.next(server)

      StreamServer.cancel(server)
    end
  end

  defmodule HTTP do
    def init(opts), do: opts

    def call(conn, {agent, owner}) do
      {:ok, request_body, conn} = Plug.Conn.read_body(conn)
      send(owner, {:request, Jason.decode!(request_body)})
      response = Agent.get_and_update(agent, fn [response | rest] -> {response, rest} end)

      {status, chunks, headers} =
        case response do
          {status, chunks} -> {status, chunks, []}
          response -> response
        end

      conn =
        conn
        |> Plug.Conn.merge_resp_headers(headers)
        |> Plug.Conn.put_resp_header("retry-after", "0")
        |> Plug.Conn.send_chunked(status)

      Enum.reduce(chunks, conn, fn chunk, conn ->
        {:ok, conn} = Plug.Conn.chunk(conn, chunk)
        conn
      end)
    end
  end

  defp response_stream(responses, opts \\ []) do
    agent = start_supervised!({Agent, fn -> responses end}, id: make_ref())
    http = start_supervised!({Bandit, plug: {HTTP, {agent, self()}}, port: 0}, id: make_ref())
    {:ok, {_address, port}} = ThousandIsland.listener_info(http)

    opts =
      Keyword.merge(
        [
          base_url: "http://127.0.0.1:#{port}",
          api_key: "test-only",
          preserve_stream_errors: true,
          receive_timeout: 5_000,
          max_retries: 0
        ],
        opts
      )

    model = ReqLLM.model!("openai:gpt-4o-mini")
    model = %{model | extra: Map.put(model.extra || %{}, :wire, %{protocol: "openai_chat"})}
    {:ok, stream} = ReqLLM.stream_text(model, "test", opts)
    stream
  end

  test "real Finch completion raises API.Stream with complete HTTP cause, one request" do
    body = Jason.encode!(@envelope)
    stream = response_stream([{422, for(<<byte <- body>>, do: <<byte>>)}])
    error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
    assert %API.Request{status: 422, response_body: @envelope} = error.cause
    assert_receive {:request, request}
    refute Map.has_key?(request, "preserve_stream_errors")
    refute_receive {:request, _}
  end

  test "real Finch empty error body does not complete successfully" do
    stream = response_stream([{401, []}])
    error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
    assert %API.Request{status: 401, response_body: ""} = error.cause
    assert error.usage == nil
  end

  test "HTTP200 SSE error after content preserves structured cause, no successful terminal or retry" do
    content = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n"
    failure = "data: " <> Jason.encode!(@envelope) <> "\n\ndata: [DONE]\n\n"
    stream = response_stream([{200, [content, failure]}], max_retries: 3)
    owner = self()

    error =
      assert_raise API.Stream, fn ->
        stream.stream |> Stream.each(&send(owner, {:chunk, &1})) |> Enum.to_list()
      end

    assert %API.StreamEvent{status: 200, response_body: @envelope} = error.cause
    assert error.usage == nil
    assert_receive {:chunk, %ReqLLM.StreamChunk{type: :content, text: "hello"}}
    refute_receive {:chunk, %ReqLLM.StreamChunk{type: :meta}}
    assert_receive {:request, _}
    refute_receive {:request, _}
  end

  for {status, kind} <- [{429, :json}, {503, :json}, {200, :json}, {503, :binary}, {200, :binary}] do
    test "HTTP #{status} #{kind} failure logs do not expose response data even from async metadata collection" do
      status = unquote(status)
      marker = "PRIVATE_PROVIDER_RESPONSE_ROUND2"

      body =
        if unquote(kind == :binary),
          do: marker,
          else: %{"error" => %{"message" => marker}, "nested" => %{"details" => marker}}

      encoded = if unquote(kind == :binary), do: body, else: Jason.encode!(body)

      chunks =
        if status == 200,
          do: [usage_event(15) <> "event: error\ndata: " <> encoded <> "\n\n"],
          else: [encoded]

      responses =
        if status == 429, do: [{status, chunks}, {status, chunks}], else: [{status, chunks}]

      log =
        capture_log([format: "$metadata$message\n", metadata: [:error_type, :http_status]], fn ->
          response = response_stream(responses, max_retries: 1)
          assert %{error: metadata_error} = MetadataHandle.await(response.metadata_handle, 5_000)
          assert metadata_error.response_body == body
          error = assert_raise API.Stream, fn -> Enum.to_list(response.stream) end
          assert error.cause == metadata_error
          assert error.cause.status == status
          assert error.cause.response_body == body

          if status == 200 do
            assert %API.StreamEvent{} = error.cause

            assert %{input_tokens: 100, output_tokens: 15, cached_tokens: 20, reasoning_tokens: 3} =
                     error.usage
          else
            assert %API.Request{} = error.cause
            assert error.usage == nil
          end

          assert_receive {:request, _}
          if status == 429, do: assert_receive({:request, _})
          refute_receive {:request, _}
        end)

      refute log =~ marker
      assert log =~ "Metadata collection failed"
      assert log =~ "http_status=#{status}"
      assert log =~ "error_type=ReqLLM.Error.API."
      if status == 429, do: assert(log =~ "Finch streaming failed")
    end
  end

  for failure <- [:raise, :exit] do
    test "metadata handle #{failure} log excludes response and exception text" do
      marker = "PRIVATE_METADATA_EXCEPTION_ROUND2"

      error =
        API.Request.exception(reason: marker, status: 503, response_body: %{"secret" => marker})

      log =
        capture_log(fn ->
          {:ok, handle} =
            MetadataHandle.start_link(fn ->
              case unquote(failure) do
                :raise -> raise error
                :exit -> exit(error)
              end
            end)

          assert %{} == MetadataHandle.await(handle, 5_000)
          GenServer.stop(handle)
        end)

      refute log =~ marker
      assert log =~ "Metadata collection"
    end
  end

  defp usage_event(output) do
    usage = %{
      "prompt_tokens" => 100,
      "completion_tokens" => output,
      "total_tokens" => 100 + output,
      "prompt_tokens_details" => %{"cached_tokens" => 20},
      "completion_tokens_details" => %{"reasoning_tokens" => 3}
    }

    "data: " <> Jason.encode!(%{"usage" => usage}) <> "\n\n"
  end

  for waiting? <- [false, true], with_usage? <- [false, true] do
    test "terminal snapshot waiting=#{waiting?} known_usage=#{with_usage?}" do
      server = start_server(preserve_stream_errors: true)
      StreamServer.http_event(server, {:status, 200})

      if unquote(with_usage?) do
        for output <- [10, 10, 15] do
          StreamServer.http_event(server, {:data, usage_event(output)})
          assert {:ok, %ReqLLM.StreamChunk{type: :meta}} = StreamServer.next(server)
        end
      end

      expected = :sys.get_state(server).metadata[:usage]
      ref = make_ref()

      if unquote(waiting?) do
        send(server, {:"$gen_call", {self(), ref}, {:next, 5_000}})
        assert [%{type: :next}] = :sys.get_state(server).waiting_callers
      end

      StreamServer.http_event(server, {:data, "data: " <> Jason.encode!(@envelope) <> "\n\n"})

      result =
        if unquote(waiting?) do
          assert_receive {^ref, reply}
          reply
        else
          StreamServer.next(server)
        end

      assert {:error, %API.StreamEvent{response_body: @envelope} = cause, ^expected} = result

      if unquote(with_usage?) do
        assert %{
                 input_tokens: 100,
                 output_tokens: 15,
                 total_tokens: 115,
                 cached_tokens: 20,
                 reasoning_tokens: 3
               } = expected
      else
        assert expected == nil
      end

      StreamServer.http_event(server, :done)
      send(server, {:DOWN, make_ref(), :process, nil, :normal})
      send(server, {:EXIT, nil, :normal})
      assert {:error, ^cause, ^expected} = StreamServer.next(server)
      StreamServer.cancel(server)
    end
  end

  for terminator <- ["\n\n", ""] do
    test "real Finch coalesced usage and error transfers snapshot with terminator #{inspect(terminator)}" do
      usage = Enum.map_join([10, 10, 15], &usage_event/1)
      failure = "data: " <> Jason.encode!(@envelope) <> unquote(terminator)
      stream = response_stream([{200, [usage <> failure]}])
      error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end

      assert %API.StreamEvent{status: 200, response_body: @envelope} = error.cause

      assert %{
               input_tokens: 100,
               output_tokens: 15,
               total_tokens: 115,
               cached_tokens: 20,
               reasoning_tokens: 3
             } = error.usage

      refute Map.has_key?(error.cause.response_body, "usage")
      assert_receive {:request, _}
      refute_receive {:request, _}
    end
  end

  test "429 retry policy preserves only final attempt envelope" do
    first = Jason.encode!(%{"error" => %{"message" => "retry"}, "outer" => "old"})
    final = Jason.encode!(@envelope)
    stream = response_stream([{429, [first]}, {429, [final]}], max_retries: 1)
    error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
    assert %API.Request{status: 429, response_body: @envelope} = error.cause
    assert_receive {:request, _}
    assert_receive {:request, _}
    refute_receive {:request, _}
  end

  test "429 followed by success does not leak the failed attempt into the stream" do
    final = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"
    stream = response_stream([{429, [Jason.encode!(@envelope)]}, {200, [final]}], max_retries: 1)
    result = Enum.to_list(stream.stream)
    assert Enum.any?(result, &match?(%ReqLLM.StreamChunk{type: :content, text: "ok"}, &1))
    refute Enum.any?(result, &Map.has_key?(&1.metadata || %{}, :error))
    assert_receive {:request, _}
    assert_receive {:request, _}
    refute_receive {:request, _}
  end

  for payload <- [
        %{
          "type" => "response.failed",
          "response" => %{"error" => %{"message" => "denied"}},
          "outer" => "sentinel"
        },
        %{"type" => "error", "code" => "blocked", "outer" => "sentinel"},
        %{"error" => %{"code" => "no_message"}, "outer" => "sentinel"}
      ] do
    test "structured SSE error #{inspect(payload)} survives provider decoding" do
      payload = unquote(Macro.escape(payload))
      stream = response_stream([{200, ["data: " <> Jason.encode!(payload) <> "\n\n"]}])
      error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
      assert %API.StreamEvent{status: 200, response_body: ^payload} = error.cause
    end
  end

  test "successful content, tool call, usage and terminal remain unchanged with opt-in" do
    events = [
      %{"choices" => [%{"delta" => %{"content" => "hello"}}]},
      %{
        "choices" => [
          %{
            "delta" => %{
              "tool_calls" => [
                %{
                  "index" => 0,
                  "id" => "call_1",
                  "type" => "function",
                  "function" => %{"name" => "lookup", "arguments" => "{}"}
                }
              ]
            }
          }
        ]
      },
      %{"choices" => [%{"delta" => %{}, "finish_reason" => "tool_calls"}]},
      %{
        "choices" => [],
        "usage" => %{"prompt_tokens" => 2, "completion_tokens" => 3, "total_tokens" => 5}
      }
    ]

    chunks = [
      Enum.map_join(events, &("data: " <> Jason.encode!(&1) <> "\n\n")) <> "data: [DONE]\n\n"
    ]

    stream = response_stream([{200, chunks}])
    result = Enum.to_list(stream.stream)
    assert Enum.any?(result, &match?(%ReqLLM.StreamChunk{type: :content, text: "hello"}, &1))
    assert Enum.any?(result, &match?(%ReqLLM.StreamChunk{type: :tool_call}, &1))
    usage = Enum.filter(result, &(&1.type == :meta and Map.has_key?(&1.metadata, :usage)))
    assert length(usage) == 1
    baseline = response_stream([{200, chunks}], preserve_stream_errors: false)
    assert Enum.to_list(baseline.stream) == result
    assert_receive {:request, _}
    assert_receive {:request, _}
    refute_receive {:request, _}
  end

  test "Finch retains raw gzip bytes and matching content encoding, without decompression" do
    compressed = :zlib.gzip(Jason.encode!(@envelope))
    stream = response_stream([{400, [compressed], [{"content-encoding", "gzip"}]}])
    error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
    assert %API.Request{status: 400, response_body: ^compressed, headers: headers} = error.cause
    assert {"content-encoding", "gzip"} in headers
  end

  test "named SSE error with binary data retains full data rather than successful assistant" do
    stream = response_stream([{200, ["event: error\ndata: full plaintext error\n\n"]}])
    error = assert_raise API.Stream, fn -> Enum.to_list(stream.stream) end
    assert %API.StreamEvent{status: 200, response_body: "full plaintext error"} = error.cause
  end

  test "truncated final 429 retains transport error and original retry count" do
    parent = self()
    transport = %Mint.TransportError{reason: :closed}

    fun = fn _, _, acc, callback, _ ->
      send(parent, :attempt)

      acc =
        Enum.reduce(
          [{:status, 429}, {:headers, [{"retry-after", "0"}]}, {:data, "partial"}],
          acc,
          callback
        )

      {:error, transport, acc}
    end

    callback = fn event, acc -> [event | acc] end

    assert {:error, ^transport, []} =
             ReqLLM.Streaming.Retry.stream(
               Finch.build(:post, "http://localhost/"),
               ReqLLM.Finch,
               [],
               callback,
               [max_retries: 1, preserve_stream_errors: true],
               fun
             )

    assert_receive :attempt
    assert_receive :attempt
    refute_receive :attempt
  end
end
