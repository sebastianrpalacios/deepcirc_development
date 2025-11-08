module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _30, _31, _32, _33, _34, _35;
assign _32 = ~_1;
assign _30 = ~(_0 | _2);
assign _33 = ~(_2 | _32);
assign _31 = ~(_0 | _30);
assign _34 = ~_33;
assign _35 = ~(_30 | _34);
assign _11 = _31 | _35;
endmodule
