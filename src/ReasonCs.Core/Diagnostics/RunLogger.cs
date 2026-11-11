using System.Diagnostics;

namespace ReasonCs.Core.Diagnostics;

/// <summary>
/// Minimal structured logger that writes to both the console and a timestamped
/// log file under <c>logs/</c>.  Every method is thread-safe (lock on the writer).
///
/// Log file format (one line per event):
///   2026-11-11 14:05:23.123  INFO   === Training ===
///   2026-11-11 14:05:25.474  STEP   step=0      loss=2.9158  grad_norm=1.2341  ms/step=2341
///   2026-11-11 14:10:45.234  EVAL   correct=47  total=500    accuracy=9.40%
///
/// Usage:
///   using var logger = RunLogger.Create();           // creates logs/run_20261111_140523.log
///   logger.Info("Training started");
///   logger.Step(step: 0, loss: 2.9f, gradNorm: 1.2f, msPerStep: 2341f);
///   logger.Eval(correct: 47, total: 500);
/// </summary>
public sealed class RunLogger : IDisposable
{
    // ── Fields ────────────────────────────────────────────────────────────────
    private readonly StreamWriter _file;
    private readonly Stopwatch    _clock = Stopwatch.StartNew();
    private readonly object       _lock  = new();

    public string LogPath { get; }
    public string RunId   { get; }

    // ── Constructor / factory ─────────────────────────────────────────────────

    private RunLogger(StreamWriter file, string path, string runId)
    {
        _file   = file;
        LogPath = path;
        RunId   = runId;
    }

    /// <summary>
    /// Create a new logger.  A timestamped file is created under <paramref name="logDir"/>
    /// (default: <c>logs/</c> relative to the current working directory).
    /// </summary>
    public static RunLogger Create(string logDir = "logs")
    {
        Directory.CreateDirectory(logDir);

        var runId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var path  = Path.Combine(logDir, $"run_{runId}.log");
        var file  = new StreamWriter(path, append: false) { AutoFlush = true };

        var logger = new RunLogger(file, path, runId);
        logger.Info($"=== reason.cs  run {runId} ===");
        logger.Info($"Log file: {Path.GetFullPath(path)}");
        return logger;
    }

    // ── Public logging methods ────────────────────────────────────────────────

    /// <summary>
    /// General-purpose informational message.
    /// Written to both the console and the log file.
    /// </summary>
    public void Info(string message)  => Write("INFO ", message, toConsole: true);

    /// <summary>
    /// Structured training-step metric line.
    /// Written to both the console and the log file.
    /// </summary>
    public void Step(int step, float loss, float gradNorm, float msPerStep)
    {
        var msg = $"step={step,-6}  loss={loss,7:F4}  grad_norm={gradNorm,7:F4}  ms/step={msPerStep,6:F0}";
        Write("STEP ", msg, toConsole: true);
    }

    /// <summary>
    /// Structured evaluation result.
    /// Summary line written to both console and file.
    /// </summary>
    public void Eval(int correct, int total, float accuracy)
    {
        var msg = $"correct={correct}  total={total}  accuracy={accuracy:P2}";
        Write("EVAL ", msg, toConsole: true);
    }

    /// <summary>
    /// Log one prediction (prompt, expected answer, model output, correct flag).
    /// Written to the log file only — too verbose for the console.
    /// </summary>
    public void Prediction(string prompt, string expected, string got, bool correct)
    {
        var mark = correct ? "✓" : "✗";
        var msg  = $"{mark}  prompt='{prompt}'  expect='{expected}'  got='{got}'";
        Write("PRED ", msg, toConsole: false);
    }

    /// <summary>
    /// Log a key=value pair (config, hyperparameter, summary statistic).
    /// Written to both console and file.
    /// </summary>
    public void Kv(string key, object value) => Info($"{key,-20} = {value}");

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        Info($"Run complete.  Elapsed: {_clock.Elapsed:hh\\:mm\\:ss\\.fff}");
        lock (_lock) _file.Dispose();
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private void Write(string level, string message, bool toConsole)
    {
        var ts   = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff");
        var line = $"{ts}  {level}  {message}";
        lock (_lock)
        {
            _file.WriteLine(line);
        }
        if (toConsole)
            Console.WriteLine(message);
    }
}
