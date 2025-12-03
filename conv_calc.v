module conv_calc #(parameter DATA_BITS = 8)(
    input clk,
    input rst_n,
    input valid_in,
    input signed [DATA_BITS-1:0] p00, p01, p02, p10, p11, p12, p20, p21, p22,
    input signed [7:0] k00, k01, k02, k10, k11, k12, k20, k21, k22,
    output reg signed [31:0] result, // 32-bit to safe keep full precision
    output reg valid_out
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
            valid_out <= 0;
        end else if (valid_in) begin
            result <= (p00 * k00) + (p01 * k01) + (p02 * k02) +
                      (p10 * k10) + (p11 * k11) + (p12 * k12) +
                      (p20 * k20) + (p21 * k21) + (p22 * k22);
            valid_out <= 1;
        end else begin
            valid_out <= 0;
        end
    end
endmodule