%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/", "test/"],
        excluded: [~r"/_build/", ~r"/deps/", ~r"/node_modules/"]
      },
      checks: %{
        disabled: [
          # Pre-existing widespread patterns — not enforced on existing code
          {Credo.Check.Design.AliasUsage, false},
          {Credo.Check.Readability.LargeNumbers, false},
          {Credo.Check.Readability.PredicateFunctionNames, false},
          {Credo.Check.Warning.ExpensiveEmptyEnumCheck, false}
        ]
      }
    }
  ]
}
