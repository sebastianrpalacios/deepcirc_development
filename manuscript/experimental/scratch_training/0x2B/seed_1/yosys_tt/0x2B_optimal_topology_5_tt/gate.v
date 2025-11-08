module gate(_0, _1, _2, _13);
input _0, _1, _2;
output _13;
wire _18, _19, _20, _21, _22;
assign _18 = ~_1;
assign _19 = ~(_2 | _18);
assign _20 = ~(_0 | _19);
assign _21 = ~(_2 | _20);
assign _22 = ~(_18 | _20);
assign _13 = _21 | _22;
endmodule
