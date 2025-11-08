module gate(_0, _1, _2, _3, _35);
input _0, _1, _2, _3;
output _35;
wire _25, _26, _27, _28, _29, _30, _31, _32, _33, _34;
assign _28 = ~_0;
assign _26 = ~_1;
assign _27 = ~_2;
assign _25 = ~_3;
assign _31 = ~(_27 | _28);
assign _29 = ~(_25 | _28);
assign _32 = ~(_26 | _31);
assign _30 = ~(_2 | _29);
assign _33 = ~_32;
assign _34 = ~(_30 | _33);
assign _35 = _34;
endmodule
