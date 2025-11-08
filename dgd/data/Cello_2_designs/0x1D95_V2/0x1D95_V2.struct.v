module m0x1D95 (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n6_0;
	wire \$n6_1;
	wire \$n11_0;
	wire \$n9_0;
	wire \$n9_1;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n7_0;

	not (\$n10_0, \$n9_0);
	nor (\$n8_0, in1, in3);
	nor (\$n6_0, in2, in3);
	nor (\$n6_1, in2, in3);
	not (\$n7_0, \$n6_1);
	nor (\$n9_0, \$n8_0, in4);
	nor (\$n9_1, \$n8_0, in4);
	nor (\$n12_0, \$n9_1, \$n7_0);
	nor (\$n11_0, \$n10_0, \$n6_0);
	nor (out, \$n11_0, \$n12_0);

endmodule
