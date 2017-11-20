thisDirectory = If[TrueQ[StringQ[$InputFileName] && $InputFileName =!= "" && FileExistsQ[$InputFileName]],
  DirectoryName[$InputFileName],
  Directory[]
];

$rawDataFiles = <|
  "Whatever_pageable" ->FileNameJoin[{thisDirectory, "raw_data", "whatever", "cudamemcpy_pinned.json"}],
  "Whatever_pinned" ->FileNameJoin[{thisDirectory, "raw_data", "whatever", "cudamemcpy_pinned.json"}],
  "Minsky_pageable" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "cudamemcpy.json"}],
  "Minsky_pinned" -> FileNameJoin[{thisDirectory, "raw_data", "minsky", "cudamemcpy_pinned.json"}]
|>;

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

makeChart[data_] :=
  BarChart[Association[
    SortBy[KeyValueMap[
      Function[{key, val},
       key -> AssociationThread[
         Lookup[val, "machine"] ->
          Lookup[val, "bytes_per_second"]/1024^3]], data], First]],
   ChartLabels -> {Placed[First /@ Keys[data], Automatic,
      Rotate[#, 90 Degree] &], None}, ChartLegends -> Automatic,
   BarSpacing -> {Automatic, 2}, PlotTheme -> "Grid",
   ScalingFunctions -> "Log", ChartStyle -> "Rainbow"];

Export["cudaMemcpy_plot.png", makeChart[groupedData], ImageSize->1600]
