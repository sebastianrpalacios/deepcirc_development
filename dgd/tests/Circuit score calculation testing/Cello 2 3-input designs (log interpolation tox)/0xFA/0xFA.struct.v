module m0xFA (input in2, in1, in3, output out);

	wire \$new_n7__0;
	wire \$new_n6__0;
	wire \$new_n5__0;

	not (\$new_n5__0, in3);
	not (\$new_n6__0, in1);
	nor (\$new_n7__0, \$new_n6__0, \$new_n5__0);
	not (out, \$new_n7__0);

endmodule
