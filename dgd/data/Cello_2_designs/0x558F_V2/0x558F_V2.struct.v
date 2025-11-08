module m0x558F (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n6_0;
	wire \$n11_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;

	not (\$n6_0, in1);
	nor (\$n8_0, \$n6_0, in2);
	nor (\$n7_0, in4, in3);
	not (\$n9_0, \$n8_0);
	nor (\$n11_0, in1, in4);
	nor (\$n10_0, \$n9_0, \$n7_0);
	nor (out, \$n10_0, \$n11_0);

endmodule
