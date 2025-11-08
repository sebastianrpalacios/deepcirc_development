module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _13, _14, _15, _16, _17, _18;
assign _16 = ~_1;
assign _13 = ~(_0 | _2);
assign _14 = ~(_0 | _13);
assign _15 = ~(_2 | _13);
assign _17 = ~_15;
assign _18 = ~(_16 | _17);
assign _11 = _14 | _18;
endmodule
