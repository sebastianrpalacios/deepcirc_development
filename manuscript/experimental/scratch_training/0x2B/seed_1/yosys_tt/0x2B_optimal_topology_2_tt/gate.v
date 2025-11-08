module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _14, _15, _16, _17, _18;
assign _15 = ~(_1 | _2);
assign _16 = ~(_1 | _15);
assign _17 = ~(_2 | _15);
assign _18 = ~(_0 | _17);
assign _14 = ~(_16 | _18);
assign _13 = _14;
endmodule
