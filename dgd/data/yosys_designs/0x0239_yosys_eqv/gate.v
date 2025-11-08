module gate(_0, _1, _2, _3, _24);
input _0, _1, _2, _3;
output _24;
wire _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _22, _23;
assign _14 = ~_0;
assign _12 = ~_1;
assign _13 = ~_2;
assign _19 = ~(_0 | _2);
assign _15 = ~(_3 | _12);
assign _17 = ~(_13 | _14);
assign _16 = ~_15;
assign _18 = ~_17;
assign _20 = ~(_16 | _17);
assign _21 = ~(_15 | _18);
assign _22 = ~(_20 | _21);
assign _23 = ~(_19 | _22);
assign _24 = _23;
endmodule
