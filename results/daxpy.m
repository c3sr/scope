thisDirectory = If[TrueQ[StringQ[$InputFileName] && $InputFileName =!= "" && FileExistsQ[$InputFileName]],
  DirectoryName[$InputFileName],
  Directory[]
];

$rawDataFiles = <|
  "Minsky" -> <|
    "Minsky/SMT0" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_0.json"}],
    "Minsky/SMT2" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_2.json"}],
    "Minsky/SMT4" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_4.json"}],
    "Minsky/SMT8" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "daxpy_smt_8.json"}],
    "Minsky/SMT0_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt0.daxpy.json"}],
    "Minsky/SMT2_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt2.daxpy.json"}],
    "Minsky/SMT4_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt4.daxpy.json"}],
    "Minsky/SMT8_TEST" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "test.smt8.daxpy.json"}]
  |>
  ,
  "Crux" -> <|
    "Crux/SM0" -> FileNameJoin[{thisDirectory, "raw_data", "crux", "daxpy_smt_0.json"}],
    "Crux/SM2" -> FileNameJoin[{thisDirectory, "raw_data", "crux", "daxpy_smt_2.json"}]
  |>
|>;

$rawMinskyDataFiles = $rawDataFiles["Minsky"];
$rawWhateverDataFiles = $rawDataFiles["Crux"];

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
  Lookup["N"]
];

makeChart[data_] := BarChart[
  Association[
    SortBy[
      KeyValueMap[
        Function[{key, val},
          key -> AssociationThread[Lookup[val, "key"] -> Lookup[val, "bytes_per_second"] / 10^9]
        ],
        data
      ],
      First[#]&
    ]
  ],
  Frame -> True,
  FrameLabel -> {Row[{Spacer[600], "N"}], "GigaBytes/Sec (Log scale)"},
  RotateLabel -> True,
  ChartLabels -> {Placed[Keys[data], Below, Rotate[#, 90 Degree] &], None},
  ImageSize -> 640,
  LegendAppearance -> "Column",
  ChartLegends -> Placed[Automatic, Right],
  BarSpacing -> {Automatic, 1},
  PlotTheme -> "FullAxesGrid",
  GridLines -> {None, Automatic},
  ScalingFunctions -> "Log"
];


chart = makeChart[groupedData];
Print[chart];

Export[FileNameJoin[{thisDirectory, "daxpy_plot.png"}], chart, ImageSize->2400, ImageResolution->400, RasterSize -> 400]
