module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _13, _14, _16, _17, _18, _20;
assign _16 = ~_1;
assign _13 = ~(_0 | _2);
assign _14 = ~(_0 | _13);
assign _17 = ~(_13 | _16);
assign _20 = ~_17;
assign _18 = ~(_2 | _20);
assign _11 = _14 | _18;
endmodule
