import { test, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";

test("desktop waveform, missing-model safety and actual metadata empty state", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1180 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Every lead/ })).toBeVisible();
  await page
    .getByRole("link", { name: "Open ECG Workstation", exact: true })
    .click();
  await page.getByRole("button", { name: /Load synthetic demo/ }).click();
  await expect(page.locator("canvas")).toHaveCount(13);
  await expect(
    page.getByRole("button", { name: "Run research classification" }),
  ).toBeDisabled();
  await expect(page.getByText("No trained model is available.")).toBeVisible();
  await page.getByLabel("Case ID (optional)").fill("SYNTHETIC-001");
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.getByRole("heading", { name: /ECG Workstation/ }).click();
  mkdirSync("../docs/images", { recursive: true });
  await page.screenshot({
    path: "../docs/images/workstation.png",
    fullPage: true,
  });
  await page
    .getByRole("link", { name: "Model Performance", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Awaiting final evaluation" }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("mobile workstation stays within viewport and supports reduced motion", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/app/predict");
  await page.getByRole("button", { name: /Load synthetic demo/ }).click();
  await expect(page.locator("canvas")).toHaveCount(13);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../artifacts/mobile-workstation.png",
    fullPage: true,
  });
  await page.getByRole("link", { name: "Saved Cases" }).click();
  await expect(
    page.getByText("Guest analyses are not persisted."),
  ).toBeVisible();
});
