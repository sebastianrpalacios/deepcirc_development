module gate(_0, _1, _2, _3, _81);
input _0, _1, _2, _3;
output _81;
wire _68, _69, _70, _71, _72, _73, _74, _75, _76, _77, _78, _79, _80;
assign _71 = ~_0;
assign _69 = ~_1;
assign _77 = ~(_0 | _1);
assign _70 = ~_2;
assign _68 = ~_3;
assign _72 = ~(_69 | _71);
assign _78 = ~(_2 | _77);
assign _75 = ~(_69 | _70);
assign _73 = ~(_68 | _72);
assign _74 = ~_73;
assign _76 = ~(_73 | _75);
assign _79 = ~(_74 | _78);
assign _80 = ~(_76 | _79);
assign _81 = _80;
endmodule
