#!/usr/bin/env python3

import argparse
import os 
import re
import sys
import shutil
import subprocess
import socket

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))



class Generator(object):
    def __init__(self, scope_path, benchmark_filter, output_prefix=None, output_postfix=None):
        self.scope_path = scope_path
        self.benchmark_filter = benchmark_filter
        self.repetitions = None
        self.output_prefix = output_prefix
        self.output_postfix = output_postfix


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
        disallowed = '<>:"/\\|?*%. '
        s = s.replace('/', '_')
        s = s.replace(':','_')
        s = s.replace('<','-')
        s = s.replace('>','-')
        s = s.replace(' ','-')
        for c in disallowed:
            if c in s:
                print(c, s)
            assert c not in s
        return s

    def escape(s):
        return s.translate(str.maketrans({
        "(": r"\(",
        ")": r"\)",
        }))

    def create(self):
        # Get matching benchmarks
        cmd = [self.scope_path]
        cmd += ["--benchmark_list_tests=true"]
        if self.benchmark_filter:
            cmd += ['--benchmark_filter=' + self.benchmark_filter]

        out = subprocess.check_output(cmd)
        out = out.decode("utf-8")

        # generate versions of benchmark names that are safe for output files
        benchmark_output_names = {}
        for b in out.splitlines():
            benchmark_output_names[str(b)] = Generator.make_filename(str(b))

        # print commands to run each benchmark
        for benchmark in sorted([k for k in benchmark_output_names]):
            output_name = benchmark_output_names[benchmark]
            cmd = [self.scope_path]
            cmd += ['--benchmark_filter="' + Generator.escape(benchmark) + '"']

            output_path = str(output_name) + ".json"
            if self.output_prefix:
                output_path = self.output_prefix + output_path

            cmd += ['--benchmark_out="' + output_path + '"']
            if self.repetitions:
                cmd += ["--benchmark_repetitions=" + self.repetitions]
            print(" ".join(cmd))

    def run():
        parser = argparse.ArgumentParser(description='Generate script to run each benchmark in a new process.')
        parser.add_argument('--benchmark_filter', type=str,
                            help='passed to SCOPE through --benchmark_filter=')
        parser.add_argument('--scope-path', type=str,
                            help='path to scope')
        parser.add_argument('--no-use-hostname', action="store_true", help="don't prefix output with hostname")
        args = parser.parse_args()

        if args.no_use_hostname:
            g = Generator(args.scope_path, args.benchmark_filter)
        else:
            g = Generator(args.scope_path, args.benchmark_filter, output_prefix=socket.gethostname() + "_")
        g.find_scope()
        g.create()


if __name__ == '__main__':

    header = """#! /bin/bash
set -xeuo pipefail
"""
    footer = """"""

    print(header)
    Generator.run()
    print(footer)
