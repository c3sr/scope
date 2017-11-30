thisDirectory = If[TrueQ[StringQ[$InputFileName] && $InputFileName =!= "" && FileExistsQ[$InputFileName]],
  DirectoryName[$InputFileName],
  Directory[]
];

$rawDataFiles = <|
  "Minsky" -> <|
    "Minsky/SMT0" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_0.json"}],
    "Minsky/SMT2" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_2.json"}],
    "Minsky/SMT4" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_4.json"}],
    "Minsky/SMT8" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_8.json"}]
  |>
  (* ,
  "Whatever" -> <|
    "SMTAll" -> FileNameJoin[{thisDirectory, "raw_data", "whatever", "with_smt.json"}],
    "NoSMTAll" -> FileNameJoin[{thisDirectory, "raw_data", "whatever", "without_smt.json"}]
  |> *)
|>;

$rawMinskyDataFiles = $rawDataFiles["Minsky"];
$rawWhateverDataFiles = $rawDataFiles["Whatever"];

$machine = "Minsky";
$rawMachineDataFiles = $rawDataFiles[$machine];

data = Table[
  rawDataFile = $rawMachineDataFiles[key];
  Module[{info},
    info = Import[rawDataFile, "RAWJSON"];
    info["benchmarks"] = Append[#, "key" -> key]& /@ info["benchmarks"];
    Append[info, "name" -> key]
  ]
  ,
  {key, Keys[$rawMachineDataFiles]}
];

groupedData = GroupBy[
  Flatten[Lookup[data, "benchmarks"]],
  Lookup["N"]
];

makeChart[data_] := BarChart[
  Association[
    SortBy[
      KeyValueMap[
        Function[{key, val},
          key -> AssociationThread[Lookup[val, "key"] -> Lookup[val, "cpu_time"] / 10^6]
        ],
        data
      ],
      First[#]&
    ]
  ],
  AxesLabel -> {"N", "GFlops"},
  ChartLabels -> {Placed[Keys[data], Automatic, Rotate[#, 90 Degree] &], None},
  ChartLegends -> Automatic,
  BarSpacing -> {Automatic, 1},
  PlotTheme -> "Grid",
  ScalingFunctions -> "Log"
];

chart = makeChart[groupedData];
Print[chart];

Export[FileNameJoin[{thisDirectory, $machine <> "_daxpy_plot.png"}], chart, ImageSize->2400]
