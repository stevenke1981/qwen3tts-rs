// Isolated process fixture for the Windows launcher's exit-code contract.
class LauncherProbe {
    static int Main(string[] args) {
        if (args.Length != 1 || args[0] != "--probe-cuda") return 99;
        return System.Int32.Parse(System.Environment.GetEnvironmentVariable("QWEN_LAUNCHER_TEST_EXITCODE"));
    }
}
