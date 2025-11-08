module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _20, _21, _22, _26, _27, _28;
assign _26 = ~(_1 | _2);
assign _28 = ~(_0 | _2);
assign _27 = ~(_26 | _28);
assign _20 = ~_27;
assign _21 = ~(_0 | _20);
assign _22 = ~(_2 | _20);
assign _11 = _21 | _22;
endmodule
