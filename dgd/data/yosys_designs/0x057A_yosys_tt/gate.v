module gate(_0, _1, _2, _3, _67);
input _0, _1, _2, _3;
output _67;
wire _53, _54, _55, _56, _57, _58, _59, _60, _61, _62, _63, _64;
assign _55 = ~_0;
assign _54 = ~_1;
assign _53 = ~_3;
assign _59 = ~(_1 | _3);
assign _56 = ~(_53 | _54);
assign _60 = ~_59;
assign _57 = ~_56;
assign _62 = ~(_55 | _56);
assign _61 = ~(_2 | _60);
assign _58 = ~(_0 | _57);
assign _63 = ~_62;
assign _64 = ~(_61 | _63);
assign _67 = _58 | _64;
endmodule
