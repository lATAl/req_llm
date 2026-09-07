defmodule ReqLLM.Streaming.TimeoutUsageTest do
  use ExUnit.Case, async: false

  alias ReqLLM.Error.API
  alias ReqLLM.StreamServer
  import ReqLLM.Test.StreamServerHelpers, only: [start_server: 1]

  @event "data: " <>
           Jason.encode!(%{
             "choices" => [
               %{"index" => 0, "delta" => %{"role" => "assistant", "content" => "hello"}}
             ],
             "usage" => %{
               "prompt_tokens" => 100,
               "completion_tokens" => 15,
               "total_tokens" => 115,
               "prompt_tokens_details" => %{"cached_tokens" => 20},
               "completion_tokens_details" => %{"reasoning_tokens" => 3}
             }
           }) <> "\n\n"

  for preserve? <- [false, true], observed? <- [false, true] do
    test "next deadline preserves observed=#{observed?} snapshot only when opted in=#{preserve?}" do
      server =
        start_server(
          preserve_stream_errors: unquote(preserve?),
          provider_mod: ReqLLM.Providers.Groq
        )

      StreamServer.http_event(server, {:status, 200})

      if unquote(observed?) do
        StreamServer.http_event(server, {:data, @event})
        assert {:ok, %{type: :meta}} = StreamServer.next(server)
      end

      state = :sys.get_state(server)
      assert :queue.is_empty(state.queue)
      snapshot = state.metadata[:usage]

      if unquote(observed?) do
        assert %{input_tokens: 100, output_tokens: 15, cached_tokens: 20, reasoning_tokens: 3} =
                 snapshot
      else
        assert snapshot == nil
      end

      expected =
        if unquote(preserve?), do: {:error, :timeout, snapshot}, else: {:error, :timeout}

      assert StreamServer.next(server, 0) == expected
      assert StreamServer.await_metadata(server, 0) == {:error, :timeout}
      assert :sys.get_state(server).metadata[:usage] == snapshot

      transport = %Finch.TransportError{reason: :timeout}
      StreamServer.http_event(server, {:error, transport})

      expected =
        if unquote(preserve?), do: {:error, transport, snapshot}, else: {:error, transport}

      assert StreamServer.next(server) == expected
      StreamServer.cancel(server)
    end
  end

  for iteration <- 1..10 do
    test "real TCP Groq content plus usage survives either receive deadline winner #{iteration}" do
      {:ok, listener} = :gen_tcp.listen(0, [:binary, active: false, ip: {127, 0, 0, 1}])
      {:ok, port} = :inet.port(listener)
      on_exit(fn -> :gen_tcp.close(listener) end)
      supervisor = start_supervised!({Task.Supervisor, []})

      task =
        Task.Supervisor.async_nolink(supervisor, fn ->
          {:ok, socket} = :gen_tcp.accept(listener)
          {:ok, _} = :gen_tcp.recv(socket, 0, 5_000)

          :ok =
            :gen_tcp.send(socket, [
              "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
              Integer.to_string(byte_size(@event), 16),
              "\r\n",
              @event,
              "\r\n"
            ])

          receive do
            :close -> :gen_tcp.close(socket)
          end
        end)

      {:ok, response} =
        ReqLLM.stream_text("groq:llama-3.3-70b-versatile", "synthetic test prompt",
          base_url: "http://127.0.0.1:#{port}",
          api_key: "test-key",
          preserve_stream_errors: true,
          receive_timeout: 100,
          max_retries: 0
        )

      owner = self()

      error =
        assert_raise API.Stream, fn ->
          response.stream |> Stream.each(&send(owner, {:chunk, &1})) |> Enum.to_list()
        end

      send(task.pid, :close)
      Task.await(task)
      assert_receive {:chunk, %{type: :meta, metadata: %{usage: %{input_tokens: 100}}}}

      assert error.cause == :timeout or
               match?(%Finch.TransportError{reason: :timeout}, error.cause)

      assert %{input_tokens: 100, output_tokens: 15, cached_tokens: 20, reasoning_tokens: 3} =
               error.usage
    end
  end
end
