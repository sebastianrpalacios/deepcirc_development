module m0x49 (input in2, in1, in3, output out);

	wire \$n14_0;
	wire \$n15_0;
	wire \$n10_0;
	wire \$n11_0;
	wire \$n12_0;
	wire \$n13_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n8_1;
	wire \$n7_0;
	wire \$n6_0;
	wire \$n5_0;

	not (\$n5_0, in2);
	not (\$n7_0, in1);
	nor (\$n11_0, in3, \$n5_0);
	nor (\$n12_0, \$n11_0, \$n7_0);
	not (\$n13_0, \$n12_0);
	not (\$n6_0, in3);
	nor (\$n8_0, in2, \$n6_0);
	nor (\$n8_1, in2, \$n6_0);
	not (\$n9_0, \$n8_0);
	nor (\$n10_0, in1, \$n9_0);
	nor (\$n14_0, \$n13_0, \$n8_1);
	nor (\$n15_0, \$n14_0, \$n10_0);
	not (out, \$n15_0);

endmodule
