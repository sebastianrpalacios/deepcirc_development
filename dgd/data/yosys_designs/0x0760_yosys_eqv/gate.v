module gate(_0, _1, _2, _3, _94);
input _0, _1, _2, _3;
output _94;
wire _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, _92, _93;
assign _85 = ~_0;
assign _83 = ~_1;
assign _84 = ~_2;
assign _82 = ~_3;
assign _89 = ~(_2 | _3);
assign _90 = ~(_83 | _85);
assign _86 = ~(_82 | _84);
assign _91 = ~(_89 | _90);
assign _87 = ~(_85 | _86);
assign _92 = ~_91;
assign _88 = ~(_1 | _87);
assign _93 = ~(_88 | _92);
assign _94 = _93;
endmodule
