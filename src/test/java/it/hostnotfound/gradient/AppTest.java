package it.hostnotfound.gradient;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Unit test
 */
public class AppTest {
    private final ByteArrayOutputStream outContent = new ByteArrayOutputStream();
    private final ByteArrayOutputStream errContent = new ByteArrayOutputStream();
    private final PrintStream originalOut = System.out;
    private final PrintStream originalErr = System.err;

    @BeforeEach
    public void setUpStreams() {
        System.setOut(new PrintStream(outContent));
        System.setErr(new PrintStream(errContent));
    }

    @AfterEach
    public void restoreStreams() {
        System.setOut(originalOut);
        System.setErr(originalErr);
    }

    private static Stream<Arguments> invalidArgumentsProvider() {
        return Stream.of(Arguments.of((Object) new String[] {}), Arguments.of((Object) new String[] { "invalid" }),
                Arguments.of((Object) new String[] { "predict" }));
    }

    @ParameterizedTest
    @MethodSource("invalidArgumentsProvider")
    @DisplayName("test invalid arguments")
    public void invalidArguments(String[] args) {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new App(args) {
                @Override
                protected void init(String op, int predictItem) {
                    // stub
                }
            };
        });
        assertTrue(outContent.toString().contains("Gradient descent using MNIST database"));
        assertEquals(exception.getMessage(), "invalid arguments");
    }

    @Test
    @DisplayName("test invalid predict argument")
    public void otherInvalidPredictArgument() {
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            new App(new String[] { "predict", "not_an_integer" }) {
                @Override
                protected void init(String op, int predictItem) {
                    // stub
                }
            };
        });
        assertTrue(outContent.toString().contains("Gradient descent using MNIST database"));
        assertEquals(exception.getMessage(), "provided argument 'not_an_integer' must be an integer.");
    }

}
