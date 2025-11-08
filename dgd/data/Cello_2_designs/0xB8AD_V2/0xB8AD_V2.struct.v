module m0xB8AD (input in2, in1, in4, in3, output out);

	wire \$n10_0;
	wire \$n10_1;
	wire \$n11_0;
	wire \$n14_0;
	wire \$n15_0;
	wire \$n9_0;
	wire \$n9_1;
	wire \$n12_0;
	wire \$n8_0;
	wire \$n8_1;
	wire \$n13_0;
	wire \$n7_0;
	wire \$n18_0;
	wire \$n16_0;
	wire \$n17_0;
	wire \$n6_0;
	wire \$n6_1;

	nor (\$n11_0, \$n8_1, in2);
	not (\$n12_0, \$n11_0);
	not (\$n8_0, in3);
	not (\$n8_1, in3);
	nor (\$n10_0, \$n9_0, \$n6_0);
	nor (\$n10_1, \$n9_0, \$n6_0);
	not (\$n7_0, in2);
	not (\$n9_0, in1);
	not (\$n9_1, in1);
	not (\$n6_0, in4);
	not (\$n6_1, in4);
	nor (\$n15_0, \$n9_1, \$n7_0);
	nor (\$n16_0, \$n15_0, \$n6_1);
	nor (\$n13_0, \$n10_1, \$n8_0);
	nor (\$n17_0, \$n13_0, \$n16_0);
	nor (\$n14_0, \$n10_0, \$n12_0);
	nor (\$n18_0, \$n14_0, \$n17_0);
	not (out, \$n18_0);

endmodule
