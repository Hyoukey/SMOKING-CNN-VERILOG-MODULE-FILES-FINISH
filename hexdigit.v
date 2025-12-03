module hexdigit (
    input clk,
    input rst_n,
    input valid_in,
    input wire [2:0] decision, // 0=Non-Smoking, 1=Smoking
    output reg [6:0] hex,      // 7-segment output (Active Low)
    output reg valid_out
);

    reg locked; // Memory bit to freeze the result

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            hex <= 7'b1111111; // Display OFF (All 1s for Active Low)
            valid_out <= 0;
            locked <= 0;       // Unlock on reset
        end else begin
            
            // Only update IF we have valid data AND we haven't locked yet
            if (valid_in == 1 && locked == 0) begin
                
                case (decision)
                    3'd1 : hex <= 7'b1000000; // Display '0' (Non-Smoking)
                    3'd0 : hex <= 7'b1111001; // Display '1' (Smoking)
                    default : hex <= 7'b0000110; // Display 'E' (Error)
                endcase
                
                valid_out <= 1; // Signal that we are finished
                locked <= 1;    // FREEZE THE DISPLAY!
            end
        end
    end

endmodule