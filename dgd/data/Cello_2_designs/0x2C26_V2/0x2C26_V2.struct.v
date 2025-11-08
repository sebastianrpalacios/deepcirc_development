module m0x2C26 (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n6_0;
	wire \$n6_1;
	wire \$n11_0;
	wire \$n9_0;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n8_1;
	wire \$n7_0;

	not (\$n9_0, \$n8_0);
	not (\$n6_0, in2);
	not (\$n6_1, in2);
	nor (\$n7_0, \$n6_1, in1);
	nor (\$n8_0, \$n7_0, in4);
	nor (\$n8_1, \$n7_0, in4);
	nor (\$n11_0, \$n6_0, in3);
	nor (\$n12_0, \$n11_0, \$n8_1);
	nor (\$n10_0, \$n9_0, in3);
	nor (out, \$n10_0, \$n12_0);

endmodule
