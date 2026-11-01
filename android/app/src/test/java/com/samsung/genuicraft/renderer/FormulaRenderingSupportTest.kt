package com.samsung.genuicraft.renderer

import androidx.compose.ui.text.style.BaselineShift
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FormulaRenderingSupportTest {
    @Test
    fun summationKeepsBothLimitsAndTermSubscripts() {
        val rendered = formulaAnnotatedString("""T = \sum_{i=1}^{n} (B_i \times R_i)""")
        assertEquals("T = ∑i=1n (Bi × Ri)", rendered.text)
        val scripts = rendered.spanStyles.filter { it.item.baselineShift != null }
        assertEquals(listOf("i=1", "n", "i", "i"), scripts.map { rendered.text.substring(it.start, it.end) })
        assertEquals(BaselineShift.Superscript, scripts[1].item.baselineShift)
        assertEquals(BaselineShift.Subscript, scripts[0].item.baselineShift)
    }

    @Test
    fun exactCommandsDoNotCorruptArrowOrUnknownCommandPrefixes() {
        assertEquals("x → y", normalizeFormulaText("""x \rightarrow y"""))
        assertEquals("""\rightward{x} + \sumthing{y}""",
            formulaAnnotatedString("""\rightward{x} + \sumthing{y}""").text)
    }

    @Test
    fun currencySpacingUnitsAndApproximationKeepTheirValues() {
        val raw = """\$18{,}400 \times 0.9235 = \$16{,}992.40 \;\rightarrow\; \text{SE tax} \approx \$2{,}600"""
        assertEquals("$18,400 × 0.9235 = $16,992.40 → SE tax ≈ $2,600", formulaAnnotatedString(raw).text)
        assertEquals("BMI=72 / 2.8224≈25.5 kg/m2",
            formulaAnnotatedString("""\mathrm{BMI}=72\,/\,2.8224\approx25.5\ \mathrm{kg/m}^{2}""").text)
        assertEquals("Height=1.68 m → 1.682=2.8224 m2",
            formulaAnnotatedString("""\text{Height}=1.68\ \mathrm{m}\ \rightarrow\ 1.68^{2}=2.8224\ \mathrm{m}^{2}""").text)
    }

    @Test
    fun alignedEquationsKeepAllRowsNumbersSignsAndUnits() {
        val raw = """\begin{aligned}\text{Female: BMR}&=10\times\text{weight}+6.25\times\text{height}-5\times\text{age}-161\\&=10\times72+6.25\times168-5\times34-161\\&=720+1{,}050-170-161\\&=1{,}439\ \text{kcal/day}\end{aligned}"""
        val normalized = normalizeFormulaText(raw)
        assertEquals(4, normalized.lines().size)
        assertEquals(normalized, normalizeFormulaText(normalized))
        assertEquals(listOf(
            "Female: BMR =10×weight+6.25×height-5×age-161",
            "=10×72+6.25×168-5×34-161",
            "=720+1,050-170-161",
            "=1,439 kcal/day"
        ), normalized.lines().map { formulaAnnotatedString(it).text })
    }

    @Test
    fun alignedFractionsAreParsedPerDisplayRow() {
        val rows = normalizeFormulaText("""\begin{aligned}x&=\frac{1}{2}\\y&\approx\frac{3}{4}\end{aligned}""").lines()
        assertEquals(2, rows.size)
        assertEquals(FormulaVisualSegment.FractionSegment("1", "2"), parseFormulaSegments(rows[0])[1])
        assertEquals(FormulaVisualSegment.FractionSegment("3", "4"), parseFormulaSegments(rows[1])[1])
        assertEquals("y ≈", formulaDisplayTextSegment("y ≈", true))
        assertEquals("x →", formulaDisplayTextSegment("x →", true))
        assertEquals("R&D =1\n=2", normalizeFormulaText("""\begin{aligned}\text{R&D}&=1\\&=2\end{aligned}"""))
    }

    @Test
    fun nestedFractionIsExplicitGroupedQuotientAndKeepsExponent() {
        val fraction = parseFormulaSegments("""\text{PV factor} = \frac{1-\left(\frac{1+g}{1+r}\right)^n}{r-g}""")[1]
            as FormulaVisualSegment.FractionSegment
        assertEquals("1-((1+g)/(1+r))n", formulaAnnotatedString(fraction.numerator).text)
        assertEquals("r-g", formulaAnnotatedString(fraction.denominator).text)
        assertTrue(formulaAnnotatedString(fraction.numerator).spanStyles.any { it.item.baselineShift == BaselineShift.Superscript })
    }

    @Test
    fun nestedTextGroupsAndCommandsInsideScriptsKeepScriptSpan() {
        val rendered = formulaAnnotatedString("""FV_{\mathrm{base}} + x^{\alpha} = \text{USD }70{,}204""")
        assertEquals("FVbase + xα = USD 70,204", rendered.text)
        assertTrue(rendered.spanStyles.any { rendered.text.substring(it.start, it.end) == "base" && it.item.baselineShift == BaselineShift.Subscript })
        assertEquals("total {net}", normalizeFormulaText("""\text{total {net}}"""))
    }

    @Test
    fun unsupportedOrIncompleteEnvironmentsStayVisible() {
        val unsupported = """\begin{matrix}1&2\\3&4\end{matrix}"""
        assertEquals(unsupported, normalizeFormulaText(unsupported))
        assertEquals(unsupported, formulaAnnotatedString(unsupported).text)
        assertEquals("""\begin{aligned}x&=1""", normalizeFormulaText("""\begin{aligned}x&=1"""))
    }

    @Test
    fun existingMortgageFractionAndMultiplicationRemainSupported() {
        val segments = parseFormulaSegments("M = P [ i(1 + i)^n ] / [ (1 + i)^n - 1 ]")
        assertEquals("M = P ×", formulaDisplayTextSegment((segments[0] as FormulaVisualSegment.TextSegment).value, true))
        val fraction = segments[1] as FormulaVisualSegment.FractionSegment
        assertEquals("i(1 + i)n", formulaAnnotatedString(fraction.numerator).text)
        assertEquals("(1 + i)n - 1", formulaAnnotatedString(fraction.denominator).text)
    }
}
