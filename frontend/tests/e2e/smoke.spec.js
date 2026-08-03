const { test, expect } = require("@playwright/test");

test("has title", async ({ page }) => {
  await page.goto("/login");
  await expect(page).toHaveTitle(/VisionInspect AI/);
});

test("shows login and registration controls", async ({ page }) => {
  await page.goto("/login");

  await expect(page.getByRole("heading", { name: "VisionInspect AI" })).toBeVisible();
  const authenticationTabs = page.getByRole("tablist", { name: "Authentication mode" });
  await expect(authenticationTabs.getByRole("button", { name: "Login" })).toBeVisible();
  const registerTab = authenticationTabs.getByRole("button", { name: "Register" });
  await expect(async () => {
    await registerTab.click();
    await expect(page.getByLabel("Requested Role")).toBeVisible({ timeout: 1000 });
  }).toPass({ timeout: 10000 });
  await expect(page.getByLabel("Requested Role")).toBeVisible();
  await expect(page.getByRole("button", { name: "Request Account" })).toBeVisible();
});

test("redirects protected pages to login without a token", async ({ page }) => {
  await page.goto("/dashboard");

  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "VisionInspect AI" })).toBeVisible();
});
