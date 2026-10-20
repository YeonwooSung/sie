/** Validate request syntax only. Profile/source authorization belongs to the server. */
export function validateLoraSelectors(options: Record<string, unknown> | undefined): void {
  let selected: string | null | undefined;
  for (const key of ["lora_adapter", "lora", "lora_id"]) {
    const value = options?.[key];
    // Undefined is omitted by the wire encoders; null is an explicit base selection.
    if (value === undefined) continue;
    if (value !== null && (typeof value !== "string" || value.trim().length === 0)) {
      throw new TypeError(`'${key}' must be a non-empty string or null`);
    }
    if (selected !== undefined && selected !== value) {
      throw new TypeError("LoRA selector fields must agree, including explicit null values");
    }
    selected = value;
  }
}
