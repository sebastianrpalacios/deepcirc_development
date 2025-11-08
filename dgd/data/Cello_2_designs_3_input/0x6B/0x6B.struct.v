module m0x6B (input in2, in1, in3, output out);

	wire \$n10_0;
	wire \$n11_0;
	wire \$n12_0;
	wire \$n13_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;
	wire \$n7_1;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	nor (\$n10_0, in3, \$n5_0);
	nor (\$n11_0, in1, \$n10_0);
	not (\$n6_0, in3);
	nor (\$n7_0, in2, \$n6_0);
	nor (\$n7_1, in2, \$n6_0);
	not (\$n8_0, \$n7_0);
	nor (\$n9_0, in1, \$n8_0);
	nor (\$n12_0, \$n11_0, \$n7_1);
	nor (\$n13_0, \$n12_0, \$n9_0);
	not (out, \$n13_0);

endmodule
