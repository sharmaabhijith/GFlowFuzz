import os, uuid, json, subprocess, pathlib, shutil
from enum import Enum, auto

class Tool(Enum):
    GCC   = auto(); CLANG = auto(); GPP = auto()
    Z3    = auto(); CVC5  = auto()
    GO    = auto(); JAVAC = auto(); QISKIT = auto()

COV_FLAGS = {
    "gcc":  ["-fprofile-arcs", "-ftest-coverage", "-g", "-O0"],
    "g++":  ["-fprofile-arcs", "-ftest-coverage", "-g", "-O0"],
    "clang":["-fprofile-instr-generate", "-fcoverage-mapping", "-g", "-O0"],
}

cmd = [compiler, *COV_FLAGS.get(compiler, []),
       "-x", "c", "-std=c2x", filename,
       "-o", f"/tmp/out{self.CURRENT_TIME}"]


class CoverageManager:
    """
    Collects and summarises per-line coverage for a single compiler/SUT.
    profile_dir: directory where .profraw, .gcda, .exec, etc. are kept
    tool: one of the Tool enum values
    binary_path: path to the instrumented executable or script runner
    """
    def __init__(self, tool: Tool, binary_path: pathlib.Path):
        self.tool, self.bin = tool, pathlib.Path(binary_path)
        self.profile_dir = pathlib.Path("/tmp/fcov") / self.bin.stem
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._total_lines = 0

    # ---------- public API used from your fuzz loop -----------------
    def run_once(self, *argv: str, timeout: float = 2.0) -> None:
        """
        Execute the instrumented SUT on one input.
        """
        if self.tool in (Tool.GCC, Tool.GPP):
            env = os.environ.copy()
            env["GCOV_PREFIX"] = str(self.profile_dir)
            env["GCOV_PREFIX_STRIP"] = "0"
            subprocess.run([self.bin, *argv], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout)
        elif self.tool is Tool.CLANG:
            env = os.environ.copy()
            env["LLVM_PROFILE_FILE"] = str(self.profile_dir /
                                           f"{uuid.uuid4()}.profraw")
            subprocess.run([self.bin, *argv], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout)
        elif self.tool in (Tool.Z3, Tool.CVC5):
            # assume you rebuilt them with clang flags, treat like CLANG
            self.run_once(*argv, timeout=timeout)  # recursion
        elif self.tool is Tool.GO:
            env = os.environ.copy()
            env["GOCOVERDIR"] = str(self.profile_dir)
            subprocess.run([self.bin, *argv], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout)
        elif self.tool is Tool.JAVAC:           # actually the Java programme
            exec_file = self.profile_dir / f"{uuid.uuid4()}.exec"
            subprocess.run(
                ["java",
                 f"-javaagent:jacocoagent.jar=destfile={exec_file}",
                 "-jar", str(self.bin), *argv],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout
            )
        elif self.tool is Tool.QISKIT:          # pure-Python
            cov_file = self.profile_dir / ".coverage"
            subprocess.run(
                ["coverage", "run", "-a", "--data-file", str(cov_file),
                 str(self.bin), *argv],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout
            )
        else:
            raise NotImplementedError

    def update_total(self) -> int:
        """
        Merge the raw profiles that appeared since the previous call and
        return *cumulative* number of unique source lines reached.
        """
        if self.tool is Tool.CLANG:
            profraws = list(self.profile_dir.glob("*.profraw"))
            if not profraws:
                return self._total_lines
            profdata = self.profile_dir / "fuzz.profdata"
            subprocess.run(["llvm-profdata", "merge", "-sparse",
                            *map(str, profraws), "-o", str(profdata)],
                           check=True)
            out = subprocess.check_output(
                ["llvm-cov", "export", "--summary-only", str(self.bin),
                 f"--instr-profile={profdata}"],
                text=True)
            lines = json.loads(out)["data"][0]["totals"]["lines"]["count"]
            for p in profraws: p.unlink()
        elif self.tool in (Tool.GCC, Tool.GPP):
            subprocess.run(["gcov", "-b", "-o", str(self.bin.parent),
                            str(self.bin)], cwd=self.profile_dir)
            lines = sum(1 for g in self.profile_dir.glob("*.gcov")
                          for _ in open(g))
            shutil.rmtree(self.profile_dir)      # start fresh
            self.profile_dir.mkdir(exist_ok=True)
        elif self.tool is Tool.GO:
            subprocess.run(["go", "tool", "covdata", "textfmt",
                            "-i", str(self.profile_dir),
                            "-o", str(self.profile_dir / "cover.out")])
            # count lines marked as executed
            lines = sum(1 for l in open(self.profile_dir / "cover.out")
                          if l.split()[-1] != "0")
            shutil.rmtree(self.profile_dir); self.profile_dir.mkdir()
        elif self.tool is Tool.JAVAC:
            execs = list(self.profile_dir.glob("*.exec"))
            if not execs: return self._total_lines
            report = self.profile_dir / "jacoco.xml"
            subprocess.run(["java", "-jar", "jacococli.jar", "report",
                            *map(str, execs),
                            "--classfiles", "build/classes",
                            "--xml", str(report)],
                           check=True)
            # Count covered <line nr=".."
            lines = sum(1 for l in open(report) if 'covered="true"' in l)
            for e in execs: e.unlink()
        elif self.tool is Tool.QISKIT:
            cov_file = self.profile_dir / ".coverage"
            if not cov_file.exists():
                return self._total_lines
            subprocess.run(["coverage", "json", "-o",
                            str(self.profile_dir / "cov.json"),
                            "--pretty-print",
                            "--data-file", str(cov_file)], check=True)
            data = json.load(open(self.profile_dir / "cov.json"))
            lines = sum(len(f["executed_lines"]) for f in data["files"].values())
            cov_file.unlink(); (self.profile_dir / "cov.json").unlink()
        else:
            raise NotImplementedError
        self._total_lines = max(self._total_lines, lines)
        return self._total_lines


# self.cover = CoverageManager(tool=Tool.CLANG,     # or GCC, GO, …
#                              binary_path=Path(f"/tmp/out{self.CURRENT_TIME}"))
# self.prev_cov = 0
# BATCH = 200
# batch_ctr, prof_ctr = 0, 0

# while …:
#     …
#     # ========== execute candidate ================
#     self.cover.run_once(fo)        # one profile file produced
#     batch_ctr += 1
#     # ========== every BATCH inputs, convert coverage into reward ==========
#     if batch_ctr == BATCH:
#         new_cov = self.cover.update_total()
#         reward  = max(new_cov - self.prev_cov, 0)
#         self.prev_cov = new_cov
#         batch_ctr = 0

#         loss = self.oracle.compute_tb_loss(
#             log_z_sum=log_zs,
#             log_prob_sum=log_probs,
#             log_reward=torch.log1p(torch.tensor(reward, dtype=torch.float32))
#         )
#         loss.backward()
