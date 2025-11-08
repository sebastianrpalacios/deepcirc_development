module gate(_0, _1, _2, _3, _105);
input _0, _1, _2, _3;
output _105;
wire _100, _101, _102, _103, _104, _95, _96, _97, _98, _99;
assign _96 = ~_1;
assign _97 = ~_2;
assign _98 = ~(_0 | _2);
assign _95 = ~_3;
assign _101 = ~(_3 | _97);
assign _99 = ~(_96 | _98);
assign _102 = ~(_96 | _101);
assign _100 = ~(_95 | _99);
assign _103 = ~(_0 | _102);
assign _104 = ~(_100 | _103);
assign _105 = _104;
endmodule
