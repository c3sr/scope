thisDirectory = If[TrueQ[StringQ[$InputFileName] && $InputFileName =!= "" && FileExistsQ[$InputFileName]],
  DirectoryName[$InputFileName],
  Directory[]
];

$rawDataFiles = <|
  "Minsky" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "cudamemcpy.json"}],
  "Whatever" ->FileNameJoin[{thisDirectory, "raw_data", "whatever", "cudamemcpy.json"}]
|>;

$rawMinskyDataFiles = $rawDataFiles["Minsky"];
$rawWhateverDataFiles = $rawDataFiles["Whatever"];

data = KeyValueMap[
    Function[{key, val},
      Module[{info = Import[val, "RAWJSON"]},
        info["benchmarks"] = Append[#, "machine" -> key]& /@ info["benchmarks"];
        info = Append[info, "machine" -> key];
        info
      ]
    ],
    $rawDataFiles
];

groupedData = GroupBy[
  Flatten[Lookup[data, "benchmarks"]],
  Lookup[{"bytes"}]
];

(*


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
  ChartLabels -> {Placed[Keys[data], Automatic, Rotate[#, 90 Degree] &], None},
  ChartLegends -> Automatic,
  BarSpacing -> {Automatic, 1},
  PlotTheme -> "Grid",
  ScalingFunctions -> "Log"
];

Export[$machine <> "_plot.png", makeChart[Take[groupedData, UpTo[10]]], ImageSize->600]
*)
