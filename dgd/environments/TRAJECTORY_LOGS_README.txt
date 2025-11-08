Each episode writes: <output_folder>/<track>/trajectories/ep_0000.json

JSON schema:
{
  "episode": <int>,
  "track": "<trained_masked|untrained_masked|random_masked>",
  "steps": [
    {
      "step": <int>,
      "action_id": <int|null>,
      "tgt": <int|null>,
      "cut": [<int>, ...] | null,
      "cand": <node-link dict|null>,
      "repl": <node-link dict|null>,
      "graph": <node-link dict|null>,
      "energy": <float>
    },
    ...
  ]
}
