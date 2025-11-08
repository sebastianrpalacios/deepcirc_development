module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _13, _14, _15, _16, _17, _18;
assign _13 = ~_0;
assign _14 = ~_1;
assign _15 = ~(_13 | _14);
assign _16 = ~(_2 | _15);
assign _17 = ~(_0 | _16);
assign _18 = ~(_2 | _16);
assign _11 = _17 | _18;
endmodule
