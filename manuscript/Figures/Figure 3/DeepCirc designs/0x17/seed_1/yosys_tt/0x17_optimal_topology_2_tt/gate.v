module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _14, _27, _28, _29, _30, _31;
assign _27 = ~_2;
assign _28 = ~(_0 | _27);
assign _29 = ~(_0 | _28);
assign _30 = ~(_27 | _28);
assign _31 = ~(_1 | _30);
assign _14 = ~(_29 | _31);
assign _13 = _14;
endmodule
