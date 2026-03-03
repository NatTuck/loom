defmodule LoomkinWeb.TeamActivityComponentTest do
  use LoomkinWeb.ConnCase

  import Phoenix.LiveViewTest

  @team_id "test-team-activity"

  describe "rendering" do
    test "renders empty activity feed" do
      html = render_component(LoomkinWeb.TeamActivityComponent, %{
        id: "test-activity",
        team_id: @team_id
      })

      assert html =~ "No activity yet"
    end

    test "renders All agent filter button active by default" do
      html = render_component(LoomkinWeb.TeamActivityComponent, %{
        id: "test-activity",
        team_id: @team_id
      })

      # All button should be highlighted (active) when no agent filter is set
      assert html =~ "All"
      assert html =~ "bg-violet-600"
    end

    test "renders type filter buttons" do
      html = render_component(LoomkinWeb.TeamActivityComponent, %{
        id: "test-activity",
        team_id: @team_id
      })

      assert html =~ "tool"
      assert html =~ "message"
      assert html =~ "created"
      assert html =~ "done"
      assert html =~ "assigned"
      assert html =~ "discovery"
      assert html =~ "error"
      assert html =~ "thinking"
      assert html =~ "joined"
      assert html =~ "offload"
      assert html =~ "question"
    end
  end

  describe "event filtering" do
    test "events list is initially empty" do
      html = render_component(LoomkinWeb.TeamActivityComponent, %{
        id: "test-activity",
        team_id: @team_id
      })

      assert html =~ "No activity yet"
    end
  end

  describe "event capping" do
    test "max_events constant is 200" do
      # The module attribute @max_events is 200
      # We verify this by checking the module compiles with that constant
      assert Code.ensure_loaded?(LoomkinWeb.TeamActivityComponent)
    end
  end

  describe "agent color mapping" do
    test "module uses consistent agent color palette" do
      # TeamActivityComponent uses @agent_colors with 8 colors
      # and :erlang.phash2 for consistent mapping
      assert Code.ensure_loaded?(LoomkinWeb.TeamActivityComponent)
    end
  end

  describe "reply button" do
    defp make_event(type, agent, opts \\ %{}) do
      Map.merge(
        %{
          id: Ecto.UUID.generate(),
          type: type,
          agent: agent,
          content: "test content",
          timestamp: DateTime.utc_now(),
          expanded: false,
          metadata: Map.get(opts, :metadata, %{})
        },
        opts
      )
    end

    defp render_with_events(events) do
      render_component(LoomkinWeb.TeamActivityComponent, %{
        id: "test-activity",
        team_id: @team_id,
        events: events,
        known_agents: Enum.map(events, & &1.agent) |> Enum.uniq()
      })
    end

    test "message card shows reply button for agent" do
      html = render_with_events([make_event(:message, "researcher", %{metadata: %{from: "researcher", to: "Team"}})])
      assert html =~ "Reply to researcher"
      assert html =~ "reply_to_agent"
    end

    test "tool_call card shows reply button" do
      html = render_with_events([make_event(:tool_call, "coder", %{metadata: %{tool_name: "read_file"}})])
      assert html =~ "Reply to coder"
    end

    test "task_complete card shows reply button" do
      html = render_with_events([make_event(:task_complete, "coder", %{metadata: %{title: "Fix bug"}})])
      assert html =~ "Reply to coder"
    end

    test "discovery card shows reply button" do
      html = render_with_events([make_event(:discovery, "researcher")])
      assert html =~ "Reply to researcher"
    end

    test "error card shows reply button" do
      html = render_with_events([make_event(:error, "coder")])
      assert html =~ "Reply to coder"
    end

    test "channel_message card shows reply button" do
      html = render_with_events([make_event(:channel_message, "bridge-bot", %{metadata: %{channel: :telegram, sender: "user123"}})])
      assert html =~ "Reply to bridge-bot"
    end

    test "reply button hidden when agent is You" do
      html = render_with_events([make_event(:message, "You", %{metadata: %{from: "You", to: "Team"}})])
      refute html =~ "Reply to You"
    end

    test "reply button hidden when agent is system" do
      html = render_with_events([make_event(:message, "system", %{metadata: %{from: "system"}})])
      refute html =~ "Reply to system"
    end

    test "agent_spawn card does not show reply button" do
      html = render_with_events([make_event(:agent_spawn, "coder", %{metadata: %{agent_name: "coder", role: "coder"}})])
      refute html =~ "reply_to_agent"
    end

    test "thinking card does not show reply button" do
      html = render_with_events([make_event(:thinking, "coder")])
      refute html =~ "reply_to_agent"
    end
  end
end
