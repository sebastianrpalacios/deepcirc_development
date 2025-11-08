module gate(_0, _1, _2, _3, _4);
input _0, _1, _2, _3;
output _4;
wire _39, _41, _46, _48, _54, _55, _56, _57, _58, _59;
assign _46 = ~_1;
assign _48 = ~_3;
assign _39 = ~(_3 | _46);
assign _41 = ~(_1 | _48);
assign _54 = ~(_39 | _41);
assign _55 = ~(_2 | _54);
assign _56 = ~(_2 | _55);
assign _57 = ~(_54 | _55);
assign _58 = ~(_0 | _56);
assign _59 = ~(_39 | _58);
assign _4 = _57 | _59;
endmodule
