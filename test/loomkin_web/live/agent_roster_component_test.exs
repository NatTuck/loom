defmodule LoomkinWeb.AgentRosterComponentTest do
  use LoomkinWeb.ConnCase

  import Phoenix.LiveViewTest

  @team_id "test-roster-team"

  defp default_budget, do: %{spent: 0.0, limit: 10.0}

  defp make_agent(name, opts \\ %{}) do
    Map.merge(
      %{
        name: name,
        role: :researcher,
        status: :idle,
        team_id: @team_id,
        current_task: nil
      },
      opts
    )
  end

  defp render_roster(agents, opts \\ []) do
    render_component(LoomkinWeb.AgentRosterComponent, %{
      id: "test-roster",
      team_id: @team_id,
      agents: agents,
      tasks: Keyword.get(opts, :tasks, []),
      budget: Keyword.get(opts, :budget, default_budget()),
      focused_agent: Keyword.get(opts, :focused_agent, nil),
      roster_version: 1,
      channel_bindings: Keyword.get(opts, :channel_bindings, [])
    })
  end

  describe "reply button" do
    test "renders reply icon for each agent row" do
      html = render_roster([make_agent("alice"), make_agent("bob")])
      assert html =~ "Reply to alice"
      assert html =~ "Reply to bob"
    end

    test "reply button has reply_agent phx-click" do
      html = render_roster([make_agent("alice")])
      assert html =~ "phx-click=\"reply_agent\""
      assert html =~ "phx-value-agent=\"alice\""
    end

    test "reply button includes team_id" do
      html = render_roster([make_agent("alice")])
      assert html =~ "phx-value-team-id=\"#{@team_id}\""
    end
  end

  describe "rendering" do
    test "renders agent names" do
      html = render_roster([make_agent("alice"), make_agent("bob")])
      assert html =~ "alice"
      assert html =~ "bob"
    end

    test "renders empty state when no agents" do
      html = render_roster([])
      assert html =~ "No agents spawned"
    end

    test "renders budget bar" do
      html = render_roster([], budget: %{spent: 5.0, limit: 10.0})
      assert html =~ "Budget"
      assert html =~ "50.0%"
    end
  end
end
