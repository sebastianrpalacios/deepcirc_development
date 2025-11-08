module gate(_0, _1, _2, _3, _712);
input _0, _1, _2, _3;
output _712;
wire _704, _705, _706, _707, _708, _709, _710, _711;
assign _706 = ~_0;
assign _704 = ~_1;
assign _705 = ~_2;
assign _707 = ~(_0 | _704);
assign _709 = ~(_704 | _705);
assign _708 = ~(_3 | _707);
assign _710 = ~(_706 | _709);
assign _711 = ~(_708 | _710);
assign _712 = _711;
endmodule
