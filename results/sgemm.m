
thisDirectory = If[TrueQ[StringQ[$InputFileName] && $InputFileName =!= "" && FileExistsQ[$InputFileName]],
  DirectoryName[$InputFileName],
  Directory[]
];

$rawDataFiles = <|
  "Minsky" -> <|
      "Minsky/SMT0_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt0.sgemm.json"}],
       "Minsky/SMT2_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt2.sgemm.json"}],
      "Minsky/SMT4_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt4.sgemm.json"}],
      "Minsky/SMT8_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt8.sgemm.json"}],
    "Minsky/SMT0" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "sgemm_smt_0.json"}],
    "Minsky/SMT2" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "sgemm_smt_2.json"}],
    "Minsky/SMT4" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "sgemm_smt_4.json"}],
    "Minsky/SMT8" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "sgemm_smt_8.json"}]
  |>
  ,
  "Whatever" -> <|
    "Whatever/SM0" -> FileNameJoin[{thisDirectory, "raw_data", "whatever", "sgemm_smt_0.json"}],
    "Whatever/SM2" -> FileNameJoin[{thisDirectory, "raw_data", "whatever", "sgemm_smt_2.json"}]
  |>
|>;


$rawMinskyDataFiles = $rawDataFiles["Minsky"];
$rawWhateverDataFiles = $rawDataFiles["Whatever"];

$rawMachineDataFiles = Join[
  $rawWhateverDataFiles,
  $rawMinskyDataFiles
];



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
  Lookup[{"K", "M", "N"}]
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
      Fold[Times, 1, First[#]] &
    ]
  ],
  AxesLabel -> {"Matrix Dimensions", "CPU Time(s)"},
  ChartLabels -> {Placed[StringRiffle[Floor[#], "\[Times]"]& /@ Keys[data], Automatic, Rotate[#, 90 Degree] &], None},
  ChartLegends -> Automatic,
  BarSpacing -> {Automatic, 1},
  PlotTheme -> "Grid",
  ScalingFunctions -> "Log"
];


Export[FileNameJoin[{thisDirectory, "sgemm_plot.png"}], makeChart[Take[groupedData, UpTo[10]]], ImageSize->2400];
