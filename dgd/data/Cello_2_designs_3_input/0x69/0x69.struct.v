module m0x69 (input in2, in1, in3, output out);

	wire \$n14_0;
	wire \$n15_0;
	wire \$n10_0;
	wire \$n11_0;
	wire \$n12_0;
	wire \$n13_0;
	wire \$n9_0;
	wire \$n8_0;
	wire \$n7_0;
	wire \$n7_1;
	wire \$n6_0;
	wire \$n6_1;
	wire \$n5_0;

	nor (\$n11_0, in1, \$n6_1);
	nor (\$n10_0, in3, \$n7_1);
	not (\$n5_0, in2);
	nor (\$n13_0, \$n10_0, \$n11_0);
	not (\$n6_0, in3);
	not (\$n6_1, in3);
	not (\$n7_0, in1);
	not (\$n7_1, in1);
	nor (\$n8_0, in1, in3);
	nor (\$n9_0, \$n7_0, \$n6_0);
	nor (\$n12_0, \$n9_0, \$n8_0);
	nor (\$n15_0, in2, \$n12_0);
	nor (\$n14_0, \$n13_0, \$n5_0);
	nor (out, \$n14_0, \$n15_0);

endmodule
