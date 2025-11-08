module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _12, _13, _14, _15, _16, _17;
assign _13 = ~(_0 | _2);
assign _14 = ~(_2 | _13);
assign _15 = ~(_0 | _13);
assign _16 = ~(_14 | _15);
assign _17 = ~(_1 | _15);
assign _12 = ~(_16 | _17);
assign _11 = _12;
endmodule
