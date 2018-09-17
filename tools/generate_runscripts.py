import argparse
import os 
import re
import sys
import shutil
import subprocess

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))



class Generator(object):
    def __init__(self, scope_path, benchmark_filter):
        self.scope_path = scope_path
        self.benchmark_filter = benchmark_filter

    def find_scope(self):
        if self.scope_path:
            return
        elif shutil.which("scope") is not None:
            self.scope_path = "scope"
            return
        else:
            build_scope = os.path.join(SCRIPT_PATH, "..", "build", "scope")
            if os.path.isfile(build_scope):
                self.scope_path = build_scope
                return
        print("couldn't find scope")
        sys.exit(-1)
    
    def make_filename(s):
        return s.replace("/", "_")

    def create(self):
        # Get matching benchmarks
        cmd = [self.scope_path]
        cmd += ["--benchmark_list_tests=true"]
        if self.benchmark_filter:
            cmd += ['--benchmark_filter=' + self.benchmark_filter]
        print(cmd)
        out = subprocess.check_output(cmd)

        # generate versions of benchmark names that are safe for output files
        benchmark_output_names = {}
        for b in out.splitlines():
            benchmark_output_names[b] = make_filename(b)

        # print commands to run each benchmark
        for benchmark in benchmark_output_names:
            output_name = benchmark_output_names[benchmark]
            cmd = [self.scope_path]
            cmd += ["--benchmark_filter=" + benchmark]
            cmd += ["--benchmark_out=" + output_name]
            if self.repetitions:
                cmd += ["--benchmark_repetitions" = self.repetitions]
            print(cmd.join(" "))

    def run():
        parser = argparse.ArgumentParser(description='Generate script to run each benchmark in a new process.')
        parser.add_argument('--benchmark_filter', type=str, nargs=1,
                            help='passed to SCOPE through --benchmark_filter=')
        parser.add_argument('--scope-path', type=str,
                            help='path to scope')
        args = parser.parse_args()

        g = Generator(args.scope_path, args.benchmark_filter)
        g.find_scope()
        g.create()




if __name__ == '__main__':
    Generator.run()