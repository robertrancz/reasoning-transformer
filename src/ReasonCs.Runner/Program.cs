using ReasonCs.Core.Data;
using ReasonCs.Core.Transformer;
using ReasonCs.Core.Training;

// ─────────────────────────────────────────────────────────────────────────────
//  reason.cs — entry point
// ─────────────────────────────────────────────────────────────────────────────
//
//  Stages (uncomment as each is implemented):
//
//  1. Tensor + Autograd tests      → run:  dotnet test
//  2. Train base model             → uncomment TrainBaseModel()
//  3. Evaluate greedy baseline     → uncomment EvalBaseline()
//  4. MCTS reasoning               → uncomment RunMcts()
//

Console.WriteLine("reason.cs — From-Scratch Reasoning Transformer");
Console.WriteLine("================================================");
Console.WriteLine("Run `dotnet test` to execute the tensor/gradient unit tests.");

// TODO: uncomment when ready
// TrainBaseModel();
// EvalBaseline();
// RunMcts();
