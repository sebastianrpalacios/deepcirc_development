module gate(_0, _1, _2, _11);
input _0, _1, _2;
output _11;
wire _12, _17, _18, _19, _20, _21;
assign _19 = ~_0;
assign _17 = ~_1;
assign _21 = ~(_0 | _2);
assign _18 = ~(_2 | _17);
assign _20 = ~(_18 | _19);
assign _12 = ~(_20 | _21);
assign _11 = _12;
endmodule
