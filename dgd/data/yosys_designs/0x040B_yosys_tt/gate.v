module gate(_0, _1, _2, _3, _46);
input _0, _1, _2, _3;
output _46;
wire _36, _37, _38, _39, _40, _41, _42, _43, _44, _45;
assign _38 = ~_0;
assign _37 = ~_1;
assign _36 = ~_3;
assign _39 = ~(_2 | _36);
assign _40 = ~_39;
assign _42 = ~(_0 | _39);
assign _41 = ~(_38 | _40);
assign _43 = ~(_37 | _42);
assign _44 = ~_43;
assign _45 = ~(_41 | _44);
assign _46 = _45;
endmodule
