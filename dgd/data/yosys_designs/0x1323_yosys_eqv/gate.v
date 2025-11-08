module gate(_0, _1, _2, _3, _181);
input _0, _1, _2, _3;
output _181;
wire _172, _173, _174, _175, _176, _177, _178, _179, _180;
assign _174 = ~_0;
assign _173 = ~_2;
assign _172 = ~_3;
assign _175 = ~(_3 | _174);
assign _176 = ~(_0 | _172);
assign _177 = ~(_175 | _176);
assign _178 = ~_177;
assign _179 = ~(_1 | _178);
assign _180 = ~(_173 | _179);
assign _181 = _180;
endmodule
